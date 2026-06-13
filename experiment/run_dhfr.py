"""DHFR 对接优化 — 1DLS MTX 共晶, SPR 反思, score + 相互作用全记录."""

import json, logging, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "docking_evaluator/src")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)

from llm_core.providers import OpenAICompatibleProvider
from experiment.models_config import load_models
from docking_evaluator import DockingEvaluator

LEAD_SMILES = "CN(Cc1cnc2nc(N)nc(O)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O"
LEAD_NAME = "Methotrexate"
KEY_RESIDUES = ["ILE7","ALA9","LEU22","ASP27","PHE31","SER59","ILE60","THR113","PHE134","GLU30","TRP24"]

RECEPTOR_PDB = "experiment/1DLS_prepared.pdb"
SMINA_BIN, CENTER, SIZE = "smina", (30.7,16.8,-1.6), (22,22,22)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_dhfr")
os.makedirs(RUNS_DIR, exist_ok=True)

def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def parse_smiles(text):
    """从 LLM 输出中提取 SMILES — 全文匹配, 不依赖分行."""
    text = strip_think(text)
    # 优先: 找 </think> 之后的纯 SMILES 行
    after = text.split("<｜end▁of▁thinking｜><｜end▁of▁thinking｜>")[-1] if "<｜end▁of▁thinking｜> response" in text else text
    lines = after.split("\n")
    smi = []
    for line in lines:
        t = line.strip()
        if len(t) < 5: continue
        if re.search(r"[\u4e00-\u9fff]", t): continue  # 纯中文行跳过
        # 只匹配看起来像 SMILES 的行
        if re.match(r"^[A-Za-z0-9\(\)\[\]@\+\-\#\%\\\/\.\=\@\[\]]{5,}$", t):
            if t not in smi: smi.append(t)
            if len(smi) >= 3: return smi
    # Fallback: 从整个文本中提取 SMILES token
    for m in re.finditer(r'[A-Za-z][A-Za-z0-9\(\)\[\]@\+\-\#\%\\\/\.\=\@\[\]]{4,}', text):
        t = m.group(0)
        if re.search(r'[\(\)\[\]\=\#\/\\@]', t) and t not in smi:
            smi.append(t)
            if len(smi) >= 3: return smi
    return smi

print("=" * 80)
print(f"DHFR Docking — {LEAD_NAME} → 1DLS")
print(f"Key residues: {KEY_RESIDUES}")
print("3 seeds × 10 rounds × 3 molecules")
print("=" * 80)

# Reference MTX score
evaluator = DockingEvaluator(receptor_pdb=RECEPTOR_PDB, smina_bin=SMINA_BIN, center=CENTER, size=SIZE, interaction_method="prolif", n_parallel=1)
ref_df = evaluator.evaluate([LEAD_SMILES], n_top=1)
ref = ref_df.iloc[0].to_dict()
ref_score = ref["best_score"]
ref_int = ref.get("interactions","")
ref_hb = ref.get("n_hbonds",0); ref_hp = ref.get("n_hydrophobic",0)
ref_pi = ref.get("n_pi_stack",0); ref_sb = ref.get("n_salt_bridges",0)
ref_kh = sum(1 for r in KEY_RESIDUES if r in ref_int)
print(f"MTX reference: score={ref_score:.4f}, HB={ref_hb}, VdW={ref_hp}, Pi={ref_pi}, SB={ref_sb}, key hits={ref_kh}/11")
print(f"Interactions: {ref_int}")

# Save reference
with open(os.path.join(RUNS_DIR,"reference_mtx.json"),"w") as f: json.dump(ref,f,indent=2)

# LLM
models = load_models()
cfg = [m for m in models if m.model == "deepseek-v4-pro"][0]
provider = OpenAICompatibleProvider(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model, timeout=cfg.timeout)

def system_prompt():
    return (
        f"你是药物化学家。设计靶向 DHFR (PDB 1DLS) 的新分子, 优化 Smina vinardo 对接分数 (越低越好, MTX 为 {ref_score:.1f})。\n"
        f"MTX 相互作用: HB×{ref_hb}, VdW×{ref_hp}, Pi×{ref_pi}, SaltBridge×{ref_sb}\n"
        "保留与 ASP27/PHE31 的关键接触。每行一个 SMILES, 不要解释。"
    )
def fmt_mol(m):
    """完整格式化单分子反馈."""
    smi = str(m.get("smiles", "?"))[:60]
    sc = m.get("best_score", "?")
    ds = f"{sc - ref_score:+.2f}" if isinstance(sc, (int, float)) else "?"
    scores = m.get("all_scores", [])
    poses = " ".join(f"{s:.2f}" for s in (scores[:9] if scores else []))
    hb = m.get("n_hbonds", 0)
    hp = m.get("n_hydrophobic", 0)
    pi = m.get("n_pi_stack", 0)
    sb = m.get("n_salt_bridges", 0)
    inter = str(m.get("interactions", ""))
    kh = sum(1 for r in KEY_RESIDUES if r in inter)
    return (
        f"  {smi}\n"
        f"    score={sc:.4f} kcal/mol (Δ={ds} vs MTX {ref_score:.4f})\n"
        f"    9 poses: [{poses}]\n"
        f"    HBond×{hb}, VdW×{hp}, Pi-Stack×{pi}, SaltBridge×{sb}\n"
        f"    Key residue hits: {kh}/{len(KEY_RESIDUES)} — {inter[:200]}"
    )


def user_message(fbs):
    p = [
        f"靶点: DHFR (PDB 1DLS), 对接盒: {CENTER}, {SIZE}",
        f"参考配体: {LEAD_SMILES}",
        f"MTX 对接分数: {ref_score:.4f}, HB={ref_hb}, VdW={ref_hp}, Pi={ref_pi}, SB={ref_sb}",
        f"MTX 接触残基: {ref_int}",
        f"DHFR 关键残基: {', '.join(KEY_RESIDUES)}",
        f"目标: 对接分数 < {ref_score:.4f} kcal/mol (越低越好)",
    ]
    for fb in fbs:
        p.append(f"\n轮次 {fb['round']}:")
        for m in fb.get("molecules", []):
            p.append(fmt_mol(m))
    p.append(f"\n请生成 3 个可能优于 MTX 的新 SMILES 分子。")
    return "\n".join(p)
def fmt_feedback(mol):
    sc = mol.get("best_score","?")
    ds = f"{sc - ref_score:+.2f}" if isinstance(sc,(int,float)) else "?"
    return f"  {mol.get('smiles','?')}\n    score={sc} (Δ={ds}), {mol.get('interactions','?')[:200]}"

def run_seed(seed):
    fbs, best_sc, best_m = [], float("inf"), None
    for rnd in range(1,11):
        t0 = time.monotonic()
        msgs = [{"role":"system","content":system_prompt()},{"role":"user","content":user_message(fbs)}]
        resp, usage = provider.generate(msgs, max_tokens=4096, temperature=0.8)
        smi = parse_smiles(resp)
        print(f"  R{rnd}: {len(smi)} SMILES ({usage['total_tokens']}t {usage['time_s']:.0f}s)")

        rd = os.path.join(RUNS_DIR, f"seed_{seed}", f"r{rnd:02d}")
        os.makedirs(rd, exist_ok=True)
        with open(f"{rd}/prompt.json","w") as f: json.dump(msgs,f,indent=2,ensure_ascii=False)
        with open(f"{rd}/resp.txt","w") as f: f.write(resp)
        with open(f"{rd}/usage.json","w") as f: json.dump(usage,f)

        if not smi:
            fbs.append({"round":rnd,"molecules":[]}); continue

        try:
            df = evaluator.evaluate(smi, n_top=3)
        except Exception as e:
            print(f"    ERR: {e}")
            fbs.append({"round":rnd,"molecules":[]}); continue

        mols = df.to_dict("records")
        for m in mols:
            sc = m.get("best_score")
            if sc is not None and sc < best_sc: best_sc, best_m = sc, m

        with open(f"{rd}/eval.json","w") as f: json.dump(mols,f,indent=2)
        fbs.append({"round":rnd,"molecules":mols[:3]})

        for i, m in enumerate(mols):
            sc = m.get('best_score', '?')
            inter = str(m.get('interactions', ''))
            print(f"    {i+1}. {str(m.get('smiles',''))[:40]} score={sc} {inter[:80]}")
        print(f"    done {time.monotonic()-t0:.0f}s, best={best_sc:.4f}")
    return best_m

for seed in [42,123,456]:
    print(f"\n{'='*60}\n Seed {seed}\n{'='*60}")
    t0 = time.monotonic()
    best = run_seed(seed)
    print(f" done {time.monotonic()-t0:.0f}s")
    if best: print(f" Best: score={best.get('best_score'):.4f} {best.get('smiles','')[:50]}")

print(f"\nDone. Results: {RUNS_DIR}")
