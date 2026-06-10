"""单分子高密度反思实验 — 每轮只生成 1 个分子并立即反馈.

对比 batch 模式 (10 rounds × 20 candidates) 与 per-molecule 模式 (20 rounds × 1 candidate).
前者每轮批量评估后选 top 5 给反馈；后者每轮只产 1 个分子，立即 DFT 评估并反映在下一轮 prompt 中.

用法:
    # 默认模型 (deepseek-v4-flash)
    python experiment/run_permolecule.py

    # 指定模型
    python experiment/run_permolecule.py --model kimi-k2.6

    # 指定种子
    python experiment/run_permolecule.py --seeds 42,123,456
"""

import argparse
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from llm_core import LLMCore
from llm_core.providers import OpenAICompatibleProvider
from evaluator import Evaluator
from reflection_rag import RAGEngine
from experiment.models_config import load_models, run_dir_name, ModelConfig

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")
PERMOLECULE_DIR = os.path.join(RUNS_DIR, "permolecule")

DEFAULT_CONFIG = {
    "target": {"HOMO-LUMO gap": "3.0 eV"},
    "n_rounds": 20,
    "n_per_round": 1,
    "n_feedback": 1,
    "max_tokens": 16384,
    "temperature": 0.8,
    "gap_range_margin": 2.0,
    "reflection_mode": "spr",
    "use_rag": True,
    "seeds": [42, 123, 456],
}

# Per-model temperature overrides (same as run_compare.py)
MODEL_TEMPERATURE = {
    "kimi-k2.6": 1.0,
}


def run_permolecule(cfg: ModelConfig, shared_config: dict) -> dict:
    model_results = {
        "model": cfg.model,
        "provider": cfg.provider,
        "seeds": [],
        "total_wall_time_s": 0.0,
        "errors": [],
    }

    provider = OpenAICompatibleProvider(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        timeout=cfg.timeout,
    )

    t_model_start = time.monotonic()
    model_run_dir = os.path.join(PERMOLECULE_DIR, run_dir_name(cfg))
    os.makedirs(model_run_dir, exist_ok=True)

    for seed in shared_config["seeds"]:
        run_name = f"seed_{seed}"
        print(f"\n  --- [{cfg.model}] per-molecule seed={seed} ---")

        evaluator = Evaluator(n_parallel=2, use_correlation=True)
        rag = RAGEngine.fork()
        core = LLMCore(
            provider=provider,
            record_prompts=True,
            reflection_mode=shared_config["reflection_mode"],
            use_rag=shared_config["use_rag"],
        )

        try:
            t0 = time.monotonic()
            result = core.run(
                target=shared_config["target"],
                n_rounds=shared_config["n_rounds"],
                n_per_round=shared_config["n_per_round"],
                n_feedback=shared_config["n_feedback"],
                evaluator=evaluator,
                rag=rag,
                run_name=run_name,
                runs_dir=model_run_dir,
                seed=seed,
                max_tokens=shared_config["max_tokens"],
                temperature=shared_config["temperature"],
                gap_range_margin=shared_config["gap_range_margin"],
            )
            elapsed = time.monotonic() - t0

            best_gap = None
            if result["best"] and result["best"].get("dft_gap_eV") is not None:
                best_gap = result["best"]["dft_gap_eV"]

            seed_info = {
                "seed": seed,
                "wall_time_s": round(elapsed, 1),
                "best_gap_eV": best_gap,
                "best_smiles": result["best"].get("smiles") if result["best"] else None,
                "failed": result["failed"],
            }
            model_results["seeds"].append(seed_info)

            status = f"gap={best_gap:.4f}" if best_gap else "FAILED"
            print(f"    {run_name} 完成: {elapsed:.0f}s — {status}")

        except Exception as e:
            error_msg = f"seed={seed}: {e}"
            model_results["errors"].append(error_msg)
            model_results["seeds"].append({"seed": seed, "error": str(e)})
            print(f"    {run_name} ERROR: {e}")
            traceback.print_exc()

    model_results["total_wall_time_s"] = round(time.monotonic() - t_model_start, 1)
    return model_results


def print_summary(all_results: list):
    print(f"\n{'=' * 90}")
    print("Per-Molecule 单分子高密度反思实验结果汇总")
    print(f"{'=' * 90}")

    header = f"{'模型':<28} {'厂商':<10} {'Seed数':<8} {'总耗时':<12} {'最佳gap':<12} {'状态'}"
    print(header)
    print("-" * 90)

    for r in all_results:
        success_seeds = [s for s in r["seeds"] if s.get("best_gap_eV") is not None]
        best_gap = min((s["best_gap_eV"] for s in success_seeds), default=None)
        n_ok = len(success_seeds)
        n_total = len(r["seeds"])
        errors = len(r["errors"])

        gap_str = f"{best_gap:.4f} eV" if best_gap else "—"
        time_str = f"{r['total_wall_time_s']:.0f}s" if r['total_wall_time_s'] > 0 else "—"

        if errors == n_total:
            status = "✗ 全部失败"
        elif errors > 0:
            status = f"⚠ {n_ok}/{n_total} 成功, {errors} 错误"
        else:
            status = f"✓ {n_ok}/{n_total}"

        print(f"{r['model']:<28} {r['provider']:<10} {n_total:<8} {time_str:<12} {gap_str:<12} {status}")

    print("-" * 90)
    print(f"结果目录: {PERMOLECULE_DIR}")
    print(f"对比 batch 模式结果: {os.path.join(RUNS_DIR, 'compare')}")


def parse_args():
    parser = argparse.ArgumentParser(description="单分子高密度反思实验")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名（默认：deepseek-v4-flash）")
    parser.add_argument("--seeds", type=str, default=None,
                        help=f"逗号分隔的种子列表（默认：{','.join(map(str, DEFAULT_CONFIG['seeds']))}）")
    parser.add_argument("--rounds", type=int, default=None,
                        help=f"迭代轮数（默认：{DEFAULT_CONFIG['n_rounds']}）")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"LLM 温度（默认：{DEFAULT_CONFIG['temperature']}）")
    parser.add_argument("--no-rag", action="store_true",
                        help="禁用 RAG")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用模型后退出")
    return parser.parse_args()


def main():
    args = parse_args()

    all_models = load_models()

    if args.list:
        print("可用模型:")
        for m in all_models:
            print(f"  {m.model:<25} ({m.provider}) — {m.base_url}")
        return

    # Default to deepseek-v4-flash if no model specified
    if args.model:
        selected = [m for m in all_models if m.model == args.model]
        if not selected:
            print(f"错误: 未找到模型 {args.model}")
            print(f"可用: {', '.join(m.model for m in all_models)}")
            sys.exit(1)
    else:
        selected = [m for m in all_models if m.model == "deepseek-v4-flash"]
        if not selected:
            print("错误: 默认模型 deepseek-v4-flash 未在 api_llm.txt 中找到")
            sys.exit(1)

    config = dict(DEFAULT_CONFIG)
    if args.rounds is not None:
        config["n_rounds"] = args.rounds
    if args.seeds is not None:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",")]
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.no_rag:
        config["use_rag"] = False

    model_cfg = selected[0]

    # Apply temperature override if applicable
    if model_cfg.model in MODEL_TEMPERATURE:
        config["temperature"] = MODEL_TEMPERATURE[model_cfg.model]
        print(f"  温度覆盖: {DEFAULT_CONFIG['temperature']} → {MODEL_TEMPERATURE[model_cfg.model]}")

    print("=" * 90)
    print("Per-Molecule 单分子高密度反思实验")
    print(f"模型: {model_cfg.model} ({model_cfg.provider})")
    print(f"目标: {config['target']}")
    print(f"迭代: {config['n_rounds']} 轮 × {config['n_per_round']} 分子/轮 = {config['n_rounds'] * config['n_per_round']} DFT")
    print(f"种子: {config['seeds']}")
    print(f"温度: {config['temperature']}, RAG: {config['use_rag']}")
    print(f"Batch 对比: 10 rounds × 20 candidates/round = 200 DFT")
    print("=" * 90)

    print("\n[1/2] 初始化 RAG 知识库...")
    RAGEngine.build_base()
    print("  RAG 知识库就绪.\n")

    print(f"[2/1] 测试 [{model_cfg.model}]")
    print("-" * 60)
    result = run_permolecule(model_cfg, config)
    print_summary([result])


if __name__ == "__main__":
    main()
