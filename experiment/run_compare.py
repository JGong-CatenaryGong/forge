"""多模型对比实验 — 对 api_llm.txt 中的所有模型运行相同实验，对比分子设计性能.

用法:
    # 全部 6 个模型，按默认参数
    python experiments/run_compare.py

    # 只测指定模型
    python experiments/run_compare.py --models deepseek-v4-flash,kimi-k2.6

    # 自定义轮数和种子
    python experiments/run_compare.py --rounds 5 --seeds 42,123
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
COMPARE_DIR = os.path.join(RUNS_DIR, "compare")

DEFAULT_CONFIG = {
    "target": {"HOMO-LUMO gap": "3.0 eV"},
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

MODEL_TEMPERATURE = {
    "kimi-k2.6": 1.0,
}


def run_model(cfg: ModelConfig, shared_config: dict) -> dict:
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
    model_run_dir = os.path.join(COMPARE_DIR, run_dir_name(cfg))
    os.makedirs(model_run_dir, exist_ok=True)

    for seed in shared_config["seeds"]:
        run_name = f"seed_{seed}"
        print(f"\n  --- [{cfg.model}] seed={seed} ---")

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
    print("对比实验结果汇总")
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
        time_str = f"{r['total_wall_time_s']:.0f}s" if r["total_wall_time_s"] > 0 else "—"

        if errors == n_total:
            status = "✗ 全部失败"
        elif errors > 0:
            status = f"⚠ {n_ok}/{n_total} 成功, {errors} 错误"
        else:
            status = f"✓ {n_ok}/{n_total}"

        print(f"{r['model']:<28} {r['provider']:<10} {n_total:<8} {time_str:<12} {gap_str:<12} {status}")

    print("-" * 90)
    print(f"结果目录: {COMPARE_DIR}")
    print("运行 python experiments/summarize.py 查看详细对比分析")


def parse_args():
    parser = argparse.ArgumentParser(description="多模型分子设计对比实验")
    parser.add_argument("--models", type=str, default=None,
                        help="逗号分隔的模型名（默认：全部）")
    parser.add_argument("--rounds", type=int, default=None,
                        help=f"迭代轮数（默认：{DEFAULT_CONFIG['n_rounds']}）")
    parser.add_argument("--seeds", type=str, default=None,
                        help=f"逗号分隔的种子列表（默认：{','.join(map(str, DEFAULT_CONFIG['seeds']))}）")
    parser.add_argument("--per-round", type=int, default=None,
                        help=f"每轮候选数（默认：{DEFAULT_CONFIG['n_per_round']}）")
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

    selected = all_models
    if args.models:
        wanted = set(args.models.split(","))
        selected = [m for m in all_models if m.model in wanted]
        if not selected:
            print(f"错误: 未找到指定模型 {args.models}")
            print(f"可用: {', '.join(m.model for m in all_models)}")
            sys.exit(1)

    config = dict(DEFAULT_CONFIG)
    if args.rounds is not None:
        config["n_rounds"] = args.rounds
    if args.seeds is not None:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",")]
    if args.per_round is not None:
        config["n_per_round"] = args.per_round
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.no_rag:
        config["use_rag"] = False

    print("=" * 90)
    print("多模型对比实验")
    print(f"目标: {config['target']}")
    print(f"轮数: {config['n_rounds']} × {config['n_per_round']} 候选/轮")
    print(f"种子: {config['seeds']}")
    print(f"温度: {config['temperature']}, RAG: {config['use_rag']}")
    print(f"待测模型 ({len(selected)}):")
    for m in selected:
        print(f"  • {m.model} ({m.provider})")
    print("=" * 90)

    print("\n[1/2] 初始化 RAG 知识库...")
    RAGEngine.build_base()
    print("  RAG 知识库就绪.\n")

    all_results = []
    for i, model_cfg in enumerate(selected):
        print(f"[2/{len(selected)}] 测试 ({i+1}/{len(selected)}) [{model_cfg.model}]")
        print("-" * 60)
        # 应用模型特定的 temperature 覆盖
        model_config = dict(config)
        if model_cfg.model in MODEL_TEMPERATURE:
            model_config["temperature"] = MODEL_TEMPERATURE[model_cfg.model]
            print(f"  温度覆盖: {config['temperature']} → {MODEL_TEMPERATURE[model_cfg.model]}")
        result = run_model(model_cfg, model_config)
        all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
