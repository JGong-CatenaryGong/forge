"""Convergence 实验 — 3 个独立 seed, 各 10 轮, target gap=5.0 eV, deepseek-v4-pro.

API key 从 api_llm.txt 加载, 无需硬编码.
用法: python experiment/run.py
"""

import logging, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)

from llm_core import LLMCore
from llm_core.providers import OpenAICompatibleProvider
from evaluator import Evaluator
from reflection_rag import RAGEngine
from experiment.models_config import load_models

# ============================================================
CONFIG = {
    "model": "deepseek-v4-pro",
    "target": {"HOMO-LUMO gap": "5.0 eV"},
    "n_rounds": 10,
    "n_per_round": 20,
    "n_feedback": 5,
    "max_tokens": 16384,
    "temperature": 0.8,
    "gap_range_margin": 2.0,
    "reflection_mode": "spr",
    "use_rag": True,
    "seeds": [42, 123, 456],
}

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# ============================================================
print("=" * 80)
print("FORGE Convergence Experiment: 3 seeds × 10 rounds × 20 candidates")
print(f"Target: {CONFIG['target']}, Model: {CONFIG['model']}")
print(f"Reflection: {CONFIG['reflection_mode']}, RAG: {CONFIG['use_rag']}")
print("=" * 80)

# 从 api_llm.txt 加载模型配置
models = load_models()
cfg = [m for m in models if m.model == CONFIG["model"]]
if not cfg:
    print(f"错误: 模型 {CONFIG['model']} 未在 api_llm.txt 中找到")
    sys.exit(1)
model_cfg = cfg[0]

RAGEngine.build_base()

provider = OpenAICompatibleProvider(
    api_key=model_cfg.api_key,
    base_url=model_cfg.base_url,
    model=model_cfg.model,
    timeout=model_cfg.timeout,
)

evaluator = Evaluator(n_parallel=2, use_correlation=True)

# ============================================================
for seed in CONFIG["seeds"]:
    run_name = f"seed_{seed}"
    print(f"\n--- Seed {seed} ---")

    rag = RAGEngine.fork()
    core = LLMCore(
        provider=provider,
        record_prompts=True,
        reflection_mode=CONFIG["reflection_mode"],
        use_rag=CONFIG["use_rag"],
    )

    t0 = time.monotonic()
    try:
        result = core.run(
            target=CONFIG["target"],
            n_rounds=CONFIG["n_rounds"],
            n_per_round=CONFIG["n_per_round"],
            n_feedback=CONFIG["n_feedback"],
            evaluator=evaluator,
            rag=rag,
            run_name=run_name,
            runs_dir=RUNS_DIR,
            seed=seed,
            max_tokens=CONFIG["max_tokens"],
            temperature=CONFIG["temperature"],
            gap_range_margin=CONFIG["gap_range_margin"],
        )
        elapsed = time.monotonic() - t0
        print(f"  {run_name} 完成: {elapsed:.0f}s")
        if result["best"]:
            print(f"  best: {result['best'].get('smiles')}  gap={result['best'].get('dft_gap_eV'):.4f}")
    except Exception as e:
        print(f"  {run_name} ERROR: {e}")

print(f"\n{'=' * 80}")
print("实验完成, 所有数据保存于:")
print(f"  {RUNS_DIR}")
for sd in sorted(os.listdir(RUNS_DIR)):
    print(f"    {sd}/")
    seed_path = os.path.join(RUNS_DIR, sd)
    if os.path.isdir(seed_path):
        for rd in sorted(os.listdir(seed_path)):
            if rd.startswith("round_"):
                print(f"      {rd}/")
