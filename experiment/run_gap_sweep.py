"""XTB Gap 扫描实验 — 测试 LLM 在不同目标 gap 下的设计能力.

使用 SimpleLLMCore + SimpleEvaluator (纯 XTB, 无 DFT),
对 1.0, 2.0, 3.0, 4.0, 5.0 eV 各跑 3 seeds × 10 rounds × 20 candidates.

用法:
    python experiment/run_gap_sweep.py
    python experiment/run_gap_sweep.py --model deepseek-v4-flash
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _pkg in ["llm_core_simple", "simple_evaluator", "llm_core", "reflection_rag", "evaluator"]:
    _src = os.path.join(sys.path[0], _pkg, "src")
    if os.path.isdir(_src):
        sys.path.insert(1, _src)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from llm_core_simple import SimpleLLMCore
from simple_evaluator import SimpleEvaluator
from llm_core.providers import OpenAICompatibleProvider
from reflection_rag import RAGEngine
from experiment.models_config import load_models, run_dir_name

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")
SWEEP_DIR = os.path.join(RUNS_DIR, "gap_sweep")

TARGET_GAPS = [1.0, 2.0, 3.0, 4.0, 5.0]

DEFAULT_CONFIG = {
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


def run_single(core: SimpleLLMCore, target_gap: float, seed: int,
               model_dir: str) -> dict:
    """对单个目标 gap 和 seed 运行."""
    run_name = f"gap_{target_gap:.1f}/seed_{seed}"
    t0 = time.monotonic()
    try:
        result = core.run(
            target={"HOMO-LUMO gap": f"{target_gap} eV"},
            n_rounds=DEFAULT_CONFIG["n_rounds"],
            n_per_round=DEFAULT_CONFIG["n_per_round"],
            n_feedback=DEFAULT_CONFIG["n_feedback"],
            run_name=run_name,
            runs_dir=model_dir,
            seed=seed,
            max_tokens=DEFAULT_CONFIG["max_tokens"],
            temperature=DEFAULT_CONFIG["temperature"],
            gap_range_margin=DEFAULT_CONFIG["gap_range_margin"],
        )
        elapsed = time.monotonic() - t0
        best = result.get("best") or {}
        best_gap = best.get("dft_gap_eV")  # 注意: 经过字段映射后存为 dft_gap_eV
        return {
            "seed": seed,
            "best_gap_eV": best_gap,
            "best_deviation": abs(best_gap - target_gap) if best_gap else None,
            "best_smiles": best.get("smiles", ""),
            "wall_time_s": round(elapsed, 1),
            "failed": result.get("failed", True),
        }
    except Exception as e:
        return {
            "seed": seed, "error": str(e), "failed": True,
            "wall_time_s": round(time.monotonic() - t0, 1),
        }


def print_summary(all_results: dict):
    """打印统计汇总."""
    target_gaps = sorted(all_results.keys())

    print("\n" + "=" * 150)
    print("XTB Gap 扫描实验结果")
    print("=" * 150)

    header = (f"{'Target':<8} {'Min|dev|':<10} {'Med|dev|':<10} {'Mean|dev|±SE':<20} "
              f"{'BestGap':<10} {'BestSeed':<10} {'AvgWall':<10} {'Seeds':<8} {'Per-seed best |dev|'}")
    print(header)
    print("-" * 150)

    for tg in target_gaps:
        seeds_data = all_results[tg]
        devs = [s["best_deviation"] for s in seeds_data if s.get("best_deviation") is not None]

        if len(devs) < 2:
            print(f"{tg:<8.1f} {'—':<10} {'—':<10} {'—':<20} {'—':<10} {'—':<10} {'—':<10} {'0/3':<8} {'所有 seed 失败'}")
            continue

        min_dev = min(devs)
        max_dev = max(devs)
        med_dev = statistics.median(devs)
        mean_dev = statistics.mean(devs)
        stderr = statistics.stdev(devs) / (len(devs) ** 0.5)

        # 最佳 seed
        best_idx = devs.index(min_dev)
        best_seed = seeds_data[best_idx]["seed"]
        best_gap = seeds_data[best_idx]["best_gap_eV"]

        walls = [s["wall_time_s"] for s in seeds_data if s.get("wall_time_s")]
        avg_wall = statistics.mean(walls) if walls else 0

        devs_str = ", ".join(f"{d:.4f}" for d in sorted(devs))
        st = f"{len(devs)}/3"

        print(f"{tg:<8.1f} {min_dev:<10.4f} {med_dev:<10.4f} {mean_dev:.4f}±{stderr:.4f}  "
              f"{best_gap:<10.4f} s{best_seed:<9} {avg_wall:<10.0f}s {st:<8} {devs_str}")

    print("-" * 150)
    print()

    # 逐 gap 逐 seed 详情
    for tg in target_gaps:
        print(f"\n--- Target gap = {tg:.1f} eV ---")
        print(f"{'Seed':<8} {'BestGap':<10} {'|dev|':<10} {'Wall':<10} {'SMILES'}")
        print("-" * 80)
        for s in all_results[tg]:
            if "error" in s:
                print(f"{s['seed']:<8} {'❌':<10} {'—':<10} {s.get('wall_time_s',0):<10.0f}s {s['error']}")
            else:
                gap = f"{s['best_gap_eV']:.4f}" if s['best_gap_eV'] else "—"
                dev = f"{s['best_deviation']:.4f}" if s['best_deviation'] else "—"
                wt = f"{s['wall_time_s']:.0f}s" if s['wall_time_s'] else "—"
                smi = s.get('best_smiles', '')[:60]
                print(f"{s['seed']:<8} {gap:<10} {dev:<10} {wt:<10} {smi}")


def parse_args():
    parser = argparse.ArgumentParser(description="XTB Gap 扫描实验")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名 (默认 deepseek-v4-flash)")
    parser.add_argument("--seeds", type=str, default=None,
                        help=f"种子 (默认 {DEFAULT_CONFIG['seeds']})")
    parser.add_argument("--rounds", type=int, default=None,
                        help=f"轮次 (默认 {DEFAULT_CONFIG['n_rounds']})")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--gaps", type=str, default=None,
                        help="目标 gap 列表, 逗号分隔 (默认 1.0,2.0,3.0,4.0,5.0)")
    parser.add_argument("--skip-rag", action="store_true", help="禁用 RAG")
    return parser.parse_args()


def main():
    args = parse_args()

    all_models = load_models()

    # 选择模型
    if args.model:
        selected = [m for m in all_models if m.model == args.model]
        if not selected:
            print(f"错误: 未找到模型 {args.model}")
            sys.exit(1)
    else:
        selected = [m for m in all_models if m.model == "deepseek-v4-flash"]
        if not selected:
            print("错误: 默认模型 deepseek-v4-flash 未找到")
            sys.exit(1)

    model_cfg = selected[0]

    config = dict(DEFAULT_CONFIG)
    if args.rounds is not None:
        config["n_rounds"] = args.rounds
    if args.seeds is not None:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",")]
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.skip_rag:
        config["use_rag"] = False
    if args.gaps is not None:
        gaps = [float(g.strip()) for g in args.gaps.split(",")]
    else:
        gaps = TARGET_GAPS

    # 温度覆盖
    if model_cfg.model in MODEL_TEMPERATURE:
        config["temperature"] = MODEL_TEMPERATURE[model_cfg.model]
        print(f"  温度覆盖: {DEFAULT_CONFIG['temperature']} → {MODEL_TEMPERATURE[model_cfg.model]}")

    print("=" * 80)
    print("XTB Gap 扫描实验")
    print(f"模型: {model_cfg.model} ({model_cfg.provider})")
    print(f"目标 gaps: {gaps}")
    print(f"轮次: {config['n_rounds']} × {config['n_per_round']} 候选 = {config['n_rounds']*config['n_per_round']} XTB/seed")
    print(f"种子: {config['seeds']}")
    print(f"温度: {config['temperature']}, RAG: {config['use_rag']}")
    print("=" * 80)

    # 初始化
    print("\n[初始化] RAG 知识库...")
    RAGEngine.build_base()

    provider = OpenAICompatibleProvider(
        api_key=model_cfg.api_key,
        base_url=model_cfg.base_url,
        model=model_cfg.model,
        timeout=model_cfg.timeout,
    )
    evaluator = SimpleEvaluator(n_parallel=4)
    core = SimpleLLMCore(provider=provider, evaluator=evaluator,
                         use_rag=config["use_rag"])

    model_dir = os.path.join(SWEEP_DIR, run_dir_name(model_cfg))
    os.makedirs(model_dir, exist_ok=True)

    all_results: dict = {g: [] for g in gaps}

    t_total_start = time.monotonic()

    for target_gap in gaps:
        print(f"\n{'=' * 60}")
        print(f"  目标 gap = {target_gap:.1f} eV")
        print(f"{'=' * 60}")

        for seed in config["seeds"]:
            rag = RAGEngine.fork() if config["use_rag"] else None
            core.evaluator = SimpleEvaluator(n_parallel=4)

            print(f"\n  --- seed={seed} ---")
            result = run_single(core, target_gap, seed, model_dir)
            all_results[target_gap].append(result)

            status = f"gap={result['best_gap_eV']:.4f}" if result.get("best_gap_eV") else "FAILED"
            print(f"    done: {status} ({result['wall_time_s']:.0f}s)")

    t_total = time.monotonic() - t_total_start
    print(f"\n总耗时: {t_total:.0f}s")

    # 保存原始数据
    summary_path = os.path.join(model_dir, "gap_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"原始数据: {summary_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
