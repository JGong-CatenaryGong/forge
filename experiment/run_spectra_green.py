"""Spectra 测试 — 绿色, 3 seeds × 10 rounds × 3 molecules, ORCA RIJCOSX."""
import logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ["evaluator_spectra", "llm_core_spectra", "llm_core", "experiment"]:
    _src = os.path.join(sys.path[0], _p, "src") if os.path.isdir(os.path.join(sys.path[0], _p, "src")) else sys.path[0]
    if _src not in sys.path: sys.path.insert(1, _src)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
from llm_core.providers import OpenAICompatibleProvider
from evaluator_spectra import SpectraEvaluator
from llm_core_spectra import SpectraLLMCore
from models_config import load_models

TARGET_CIE = (0.15, 0.06)  # 高饱和蓝色
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_spectra")
os.makedirs(RUNS_DIR, exist_ok=True)
print("=" * 80)
print(f"FORGE — ORCA RIJCOSX — Target CIE={TARGET_CIE} (蓝色)")
print("3 seeds × 10 rounds × 3 molecules/round")
print("=" * 80)

models = load_models()
cfg = [m for m in models if m.model == "deepseek-v4-pro"]
if not cfg: print("deepseek-v4-pro not found"); sys.exit(1)
model_cfg = cfg[0]

provider = OpenAICompatibleProvider(api_key=model_cfg.api_key, base_url=model_cfg.base_url, model=model_cfg.model, timeout=model_cfg.timeout)
evaluator = SpectraEvaluator(n_parallel=3, nstates=10, functional="PBE0", basis="def2-SVP")

for seed in [42, 123, 456]:
    print(f"\n{'='*60}\n  Seed {seed}\n{'='*60}")
    core = SpectraLLMCore(provider=provider, evaluator=evaluator)
    t0 = time.monotonic()
    try:
        result = core.run(target_cie=TARGET_CIE, n_rounds=10, n_per_round=3, n_feedback=3,
                          run_name=f"green/seed_{seed}", runs_dir=RUNS_DIR, seed=seed,
                          max_tokens=8192, temperature=0.8)
        elapsed = time.monotonic() - t0
        best = result.get("best") or {}
        print(f"  Seed {seed} done: {elapsed:.0f}s")
        if best.get("smiles"):
            print(f"  best: {best['smiles']}")
            print(f"    感知CIE=({best.get('cie_x_perceived','?')}, {best.get('cie_y_perceived','?')})")
            print(f"    λ_max={best.get('lambda_max_nm','?')} nm")
    except Exception as e:
        import traceback
        print(f"  Seed {seed} ERROR: {e}"); traceback.print_exc()

print(f"\nDone. Results in {RUNS_DIR}")
