"""SA-优化实验 — 保持 S0-S1 激发能, 降低 SA Score."""

import json, logging, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ["sa_optimizer_evaluator", "evaluator_spectra", "llm_core", "experiment"]:
    _s = os.path.join(sys.path[0], _p, "src") if os.path.isdir(os.path.join(sys.path[0], _p, "src")) else sys.path[0]
    if _s not in sys.path: sys.path.insert(1, _s)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)

from llm_core.providers import OpenAICompatibleProvider
from sa_optimizer_evaluator import SAOptimizerEvaluator
from models_config import load_models

REF = json.load(open("experiment/lead_reference.json"))
LEAD = REF["smiles"]
TARGET_S1 = round(REF["s1_energy_eV"], 2)
S1_TOL = 0.3

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_sa_optim")
os.makedirs(RUNS_DIR, exist_ok=True)

print("=" * 80)
print(f"SA Optimization — Lead: {LEAD}")
print(f"Target S1: {TARGET_S1} eV (±{S1_TOL}), Lead SA: {REF['sa_score']:.4f}")
print("3 seeds × 10 rounds × 3 molecules/round")
print("=" * 80)

models = load_models()
cfg = [m for m in models if m.model == "deepseek-v4-pro"][0]
provider = OpenAICompatibleProvider(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model, timeout=cfg.timeout)
evaluator = SAOptimizerEvaluator(n_parallel=1, s1_tolerance=S1_TOL)


def build_system_prompt():
    return (
        "你是一个计算化学家。给定一个先导分子及其光谱数据，设计结构类似的分子（保持核心发色团不变），目标：\n"
        "1. S0-S1 激发能保持在目标值附近\n"
        "2. 降低 SA Score（1-10，越低越容易合成）\n"
        "每行输出一个 SMILES，不要编号、不要解释、不要额外文字。"
    )


def build_user_message(round_feedbacks):
    parts = [
        f"先导分子: {LEAD}",
        f"目标 S0-S1: {TARGET_S1} eV (±{S1_TOL}), 先导 SA={REF['sa_score']:.4f}",
    ]
    if round_feedbacks:
        parts.append("之前反馈：")
        for fb in round_feedbacks:
            parts.append(f"  轮次 {fb['round']}:")
            for m in fb.get("molecules", []):
                parts.append(f"    {m.get('smiles','?')[:50]}")
                parts.append(f"      S1={m.get('s1_energy_eV','?'):.3f}eV SA={m.get('sa_score','?'):.3f}")
    parts.append(f"\n请生成 3 个新 SMILES。")
    return "\n".join(parts)


def parse_smiles_fast(text):
    """从 LLM 输出中提取 SMILES — 拒绝中文字符."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    smiles = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        if re.search(r"[\u4e00-\u9fff]", line): continue  # 中文
        if line.startswith(("#", "//", ">", "-")): continue
        if "<" in line or ">" in line or '"' in line: continue
        m = re.match(r"^\s*(?:\d+[\.\)]\s*)?(\S+)\s*$", line)
        if m:
            t = m.group(1)
            if any(c.isalpha() for c in t) and len(t) >= 2:
                smiles.append(t)
                if len(smiles) >= 3: break
    return smiles


def run_seed(seed):
    all_feedback, best_mol, best_combined = [], None, 1e9

    for rnd in range(1, 11):
        t0 = time.monotonic()
        msgs = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_message(all_feedback)},
        ]
        resp, usage = provider.generate(msgs, max_tokens=4096, temperature=0.8)
        smi = parse_smiles_fast(resp)
        print(f"  Round {rnd}: {len(smi)} SMILES ({usage['total_tokens']} tokens, {usage['time_s']:.1f}s)")

        if not smi:
            all_feedback.append({"round": rnd, "molecules": []}); continue

        try:
            df = evaluator.evaluate(smi, target_s1=TARGET_S1, n_top=3)
        except Exception as e:
            print(f"    ERROR: {e}")
            all_feedback.append({"round": rnd, "molecules": []}); continue

        mols = df.to_dict("records")
        for m in mols:
            if m.get("orca_success"):
                s1d = abs(m.get("s1_energy_eV", 999) - TARGET_S1)
                sa = m.get("sa_score", 999)
                c = (0 if s1d <= S1_TOL else 1000) + sa
                if c < best_combined:
                    best_combined, best_mol = c, m

        all_feedback.append({"round": rnd, "molecules": mols[:3]})

        rd = os.path.join(RUNS_DIR, f"sa_seed_{seed}", f"round_{rnd:02d}")
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "eval.json"), "w") as f: json.dump(mols, f, indent=2)
        with open(os.path.join(rd, "usage.json"), "w") as f: json.dump(usage, f)

        n_ok = sum(1 for m in mols if m.get("orca_success"))
        print(f"    {n_ok}/3 OK, {time.monotonic()-t0:.0f}s — best: S1={best_mol.get('s1_energy_eV',-1):.3f}eV SA={best_mol.get('sa_score',-1):.3f}")

    return best_mol


for seed in [42, 123, 456]:
    print(f"\n{'='*60}\n  Seed {seed}\n{'='*60}")
    t0 = time.monotonic()
    best = run_seed(seed)
    print(f"  Seed {seed} done: {time.monotonic()-t0:.0f}s")
    if best: print(f"  Best: {best.get('smiles','')[:60]}\n    S1={best.get('s1_energy_eV'):.3f}eV SA={best.get('sa_score'):.3f}")

print(f"\nDone: {RUNS_DIR}")
