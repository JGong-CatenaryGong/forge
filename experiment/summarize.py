"""对比实验结果汇总 — 读取 runs/compare/ 下的各模型数据，生成排名对比.

用法:
    python experiments/summarize.py
    python experiments/summarize.py --target 3.0
    python experiments/summarize.py --dir runs/compare
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SeedResult:
    seed: int
    best_gap_eV: Optional[float]
    best_deviation_eV: Optional[float]
    wall_time_s: float
    total_tokens: int
    total_rounds: int
    dft_success: int
    dft_failed: int
    failed: bool
    error: Optional[str] = None


@dataclass
class ModelSummary:
    model: str
    seeds: List[SeedResult]
    target_gap: float

    @property
    def success_seeds(self) -> List[SeedResult]:
        return [s for s in self.seeds if s.best_gap_eV is not None]

    @property
    def best_deviation(self) -> Optional[float]:
        devs = [s.best_deviation_eV for s in self.success_seeds]
        return min(devs) if devs else None

    @property
    def best_gap(self) -> Optional[float]:
        gaps = [s.best_gap_eV for s in self.success_seeds]
        return min(gaps, key=lambda g: abs(g - self.target_gap)) if gaps else None

    @property
    def avg_wall_time_s(self) -> float:
        times = [s.wall_time_s for s in self.seeds if s.wall_time_s > 0]
        return sum(times) / len(times) if times else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.seeds)

    @property
    def total_dft_success(self) -> int:
        return sum(s.dft_success for s in self.seeds)

    @property
    def total_dft_failed(self) -> int:
        return sum(s.dft_failed for s in self.seeds)

    @property
    def dft_success_rate(self) -> float:
        total = self.total_dft_success + self.total_dft_failed
        return self.total_dft_success / total if total > 0 else 0.0

    @property
    def rank_key(self) -> tuple:
        return (
            self.best_deviation if self.best_deviation is not None else float("inf"),
            -(self.dft_success_rate),
            self.avg_wall_time_s,
        )


def load_seed_result(seed_dir: str) -> Optional[SeedResult]:
    manifest_path = os.path.join(seed_dir, "manifest.json")
    summary_path = os.path.join(seed_dir, "summary.json")

    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    totals = manifest.get("totals", {})
    best = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
            best = s.get("best_molecule")

    seed_name = os.path.basename(seed_dir)
    try:
        seed = int(seed_name.split("_")[-1])
    except (ValueError, IndexError):
        seed = -1

    best_gap = None
    if best and best.get("dft_gap_eV") is not None:
        best_gap = best["dft_gap_eV"]

    return SeedResult(
        seed=seed,
        best_gap_eV=best_gap,
        best_deviation_eV=None,
        wall_time_s=totals.get("total_wall_time_s", 0),
        total_tokens=totals.get("total_tokens", 0),
        total_rounds=manifest.get("total_rounds", 0),
        dft_success=totals.get("total_dft_success", 0),
        dft_failed=totals.get("total_dft_failed", 0),
        failed=manifest.get("run_failed", True),
    )


def collect_results(compare_dir: str, target_gap: float) -> List[ModelSummary]:
    models = []
    if not os.path.isdir(compare_dir):
        return models

    for model_name in sorted(os.listdir(compare_dir)):
        model_dir = os.path.join(compare_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        seeds = []
        for entry in sorted(os.listdir(model_dir)):
            seed_dir = os.path.join(model_dir, entry)
            if not os.path.isdir(seed_dir) or not entry.startswith("seed_"):
                continue
            sr = load_seed_result(seed_dir)
            if sr:
                if sr.best_gap_eV is not None:
                    sr.best_deviation_eV = abs(sr.best_gap_eV - target_gap)
                seeds.append(sr)

        if seeds:
            models.append(ModelSummary(model=model_name, seeds=seeds, target_gap=target_gap))

    return models


def print_table(models: List[ModelSummary], target_gap: float):
    models.sort(key=lambda m: m.rank_key)

    header = (
        f"{'排名':<5} {'模型':<30} {'最佳gap':<12} {'偏差':<10} "
        f"{'DFT成功率':<10} {'平均耗时':<12} {'总Tokens':<12} {'成功/总数'}"
    )
    sep = "-" * len(header)

    print(f"\n目标 HOMO-LUMO gap: {target_gap} eV")
    print(f"\n{header}")
    print(sep)

    for rank, m in enumerate(models, 1):
        best_gap = f"{m.best_gap:.4f}" if m.best_gap else "—"
        best_dev = f"{m.best_deviation:.4f}" if m.best_deviation is not None else "—"
        rate = f"{m.dft_success_rate:.1%}"
        avg_time = f"{m.avg_wall_time_s:.0f}s"
        tokens = f"{m.total_tokens:,}"
        seed_status = f"{len(m.success_seeds)}/{len(m.seeds)}"

        marker = ""
        if rank == 1:
            marker = " ★"
        elif rank == 2:
            marker = " ☆"

        print(
            f"{rank:<5} {m.model + marker:<30} {best_gap:<12} {best_dev:<10} "
            f"{rate:<10} {avg_time:<12} {tokens:<12} {seed_status}"
        )

    print(sep)
    print(f"共 {len(models)} 个模型完成实验\n")


def print_per_seed_detail(models: List[ModelSummary]):
    print(f"\n{'=' * 90}")
    print("各 Seed 详细数据")
    print(f"{'=' * 90}")

    for m in sorted(models, key=lambda m: m.rank_key):
        print(f"\n--- {m.model} ---")
        for s in m.seeds:
            gap = f"{s.best_gap_eV:.4f}" if s.best_gap_eV else "—"
            dev = f"{s.best_deviation_eV:.4f}" if s.best_deviation_eV is not None else "—"
            tokens = f"{s.total_tokens:,}"
            status = "✗ 失败" if s.failed else "✓"
            print(
                f"  seed={s.seed:<5} gap={gap:<10} dev={dev:<10} "
                f"DFT ok={s.dft_success:<5} time={s.wall_time_s:.0f}s  tokens={tokens}  {status}"
            )


def main():
    parser = argparse.ArgumentParser(description="多模型对比实验汇总分析")
    parser.add_argument("--dir", type=str, default=None,
                        help="对比结果目录（默认：experiments/runs/compare）")
    parser.add_argument("--target", type=float, default=3.0,
                        help="目标 HOMO-LUMO gap (eV)")
    parser.add_argument("--detail", action="store_true",
                        help="显示每个 seed 的详细数据")
    args = parser.parse_args()

    if args.dir:
        compare_dir = args.dir
    else:
        compare_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "compare"
        )

    if not os.path.isdir(compare_dir):
        print(f"目录不存在: {compare_dir}")
        print("请先运行 python experiments/run_compare.py 生成实验数据")
        sys.exit(1)

    models = collect_results(compare_dir, args.target)

    if not models:
        print(f"目录 {compare_dir} 中没有找到实验结果")
        print("请先运行 python experiments/run_compare.py 生成实验数据")
        return

    print_table(models, args.target)

    if args.detail:
        print_per_seed_detail(models)


if __name__ == "__main__":
    main()
