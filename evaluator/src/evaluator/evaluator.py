"""Evaluator — 主编排器：SMILES → XTB预筛选 → pySCF精确计算 → 结构化输出.

流程:
    1. 接收 n_input 个 SMILES 和目标 target_gap
    2. 并行/串行运行 XTB 几何优化，获取 xtb_gap
    3. 按 |xtb_gap - target_gap| 排序，取 n_top 个最优分子
    4. 对 n_top 分子运行 pySCF DFT 精确计算
    5. 返回结构化表格（pandas DataFrame）
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .molecule import MoleculeData
from .xtb_runner import XTBRunner
from .pyscf_runner import PySCFRunner

logger = logging.getLogger(__name__)


class Evaluator:
    """分子评估器：通过 XTB + pySCF 两级计算评估分子接近目标属性的程度.

    用法:
        evaluator = Evaluator()
        df = evaluator.evaluate(
            smiles_list=["CCO", "CCN", "C1CCCCC1"],
            target_gap=5.0,
            n_top=2,
        )
        print(df)
    """

    def __init__(
        self,
        xtb_bin: Optional[str] = None,
        xtb_gfn: int = 2,
        xtb_omp_threads: int = 4,
        xtb_timeout: int = 300,
        dft_functional: str = "B3LYP",
        dft_basis: str = "6-31G(2d,p)",
        dft_charge: int = 0,
        dft_spin: int = 0,
        dft_max_scf_cycles: int = 200,
        n_parallel: int = 4,
        use_correlation: bool = False,
        correlation_n_samples: int = 500,
        correlation_force_retrain: bool = False,
    ):
        """
        Args:
            xtb_bin: xtb 二进制路径.
            xtb_gfn: GFN 方法版本 (2 = GFN2-xTB).
            xtb_omp_threads: 每个 xtb 进程的 OpenMP 线程数.
            xtb_timeout: 单分子 xtb 超时(秒).
            dft_functional: DFT 泛函 (如 B3LYP, PBE0, ωB97X-D).
            dft_basis: 基组 (如 6-31G(2d,p), def2-SVP, cc-pVDZ).
            dft_charge: 分子总电荷.
            dft_spin: 自旋多重度 (0=闭壳层单重态).
            dft_max_scf_cycles: SCF 最大迭代步数.
            n_parallel: 并行度（XTB 和 pySCF 各自的最大并行分子数）.
            use_correlation: 是否启用 XTB→DFT 相关性校准.
            correlation_n_samples: 训练校准模型的 QM9 样本数.
            correlation_force_retrain: 强制重新训练校准模型.
        """
        self.xtb_runner = XTBRunner(
            xtb_bin=xtb_bin,
            gfn=xtb_gfn,
            omp_num_threads=xtb_omp_threads,
            timeout=xtb_timeout,
        )
        self.pyscf_runner = PySCFRunner(
            functional=dft_functional,
            basis=dft_basis,
            charge=dft_charge,
            spin=dft_spin,
            max_scf_cycles=dft_max_scf_cycles,
        )
        self.n_parallel = n_parallel
        self.use_correlation = use_correlation
        self._correlation_model = None
        self._correlation_n_samples = correlation_n_samples
        self._correlation_force_retrain = correlation_force_retrain

    def evaluate(
        self,
        smiles_list: List[str],
        target_gap: float,
        n_top: int,
        work_dir: Optional[str] = None,
        sort_by: str = "gap",
    ) -> pd.DataFrame:
        """执行完整的评估流程.
        Args:
            smiles_list: 候选 SMILES 列表.
            target_gap: 目标 HOMO-LUMO gap (eV).
            n_top: 进入 pySCF 精确计算的分子数（XTB 偏差最小的前 n_top 个）.
            work_dir: 工作目录, None 则使用临时目录.
            sort_by: 排序依据 ("gap" = HOMO-LUMO gap).
        Returns:
            pandas DataFrame，包含 n_top 个分子的所有计算属性.
        """
        logger.info(f"开始评估: {len(smiles_list)} 个SMILES, target_gap={target_gap}eV, n_top={n_top}"
                     f"{', correlation=ON' if self.use_correlation else ''}")
        # ---- 阶段 1: XTB 几何优化 ----
        molecules = [MoleculeData(smiles=s) for s in smiles_list]
        logger.info(f"阶段1: XTB 几何优化 ({len(molecules)} 个分子, 并行度={self.n_parallel})")
        molecules = self._run_xtb_parallel(molecules, work_dir)
        n_success = sum(1 for m in molecules if m.xtb_success)
        n_failed = len(molecules) - n_success
        if n_failed > 0:
            logger.warning(f"XTB 阶段: {n_success} 成功, {n_failed} 失败")
        if n_success == 0:
            raise RuntimeError("所有分子的 XTB 计算均失败，无法继续")
        # ---- 阶段 1.5: 相关性校准 (可选) ----
        if self.use_correlation:
            self._ensure_correlation_model()
            for m in molecules:
                if m.xtb_success and m.xtb_homo is not None and m.xtb_lumo is not None and m.xtb_gap is not None:
                    calib = self._correlation_model.calibrate(
                        m.xtb_homo, m.xtb_lumo, m.xtb_gap, smiles=m.smiles
                    )
                    m.calib_homo = calib["homo"]
                    m.calib_lumo = calib["lumo"]
                    m.calib_gap = calib["gap"]
        for m in molecules:
            if m.xtb_success:
                gap_for_sort = m.calib_gap if (self.use_correlation and m.calib_gap is not None) else m.xtb_gap
                if gap_for_sort is not None:
                    m.xtb_deviation = abs(gap_for_sort - target_gap)
        molecules.sort(
            key=lambda m: m.xtb_deviation if m.xtb_deviation is not None else float("inf")
        )
        top_molecules = molecules[:n_top]
        top_molecules = [m for m in top_molecules if m.xtb_success]
        sort_label = "calib" if self.use_correlation else "XTB"
        logger.info(
            f"阶段2: 排序完成, top {len(top_molecules)} 分子 "
            f"{sort_label} gap 范围: [{top_molecules[0].xtb_gap:.2f} .. {top_molecules[-1].xtb_gap:.2f}] eV"
        )
        if len(top_molecules) == 0:
            raise RuntimeError("没有成功完成 XTB 的分子可用于 DFT 计算")
        # ---- 阶段 3: pySCF DFT 精确计算 ----
        logger.info(f"阶段3: pySCF DFT 计算 ({len(top_molecules)} 个分子)")
        top_molecules = self._run_pyscf_parallel(top_molecules)
        # ---- 阶段 4: 构建输出 ----
        return self._build_output(top_molecules, target_gap)

    def _run_xtb_parallel(
        self, molecules: List[MoleculeData], work_dir: Optional[str] = None
    ) -> List[MoleculeData]:
        """并行运行 XTB 计算."""
        # XTB 是 I/O 密集型 (子进程) + CPU密集型 (OpenMP)，串行更简单可靠
        # 但如果 n_parallel > 1，使用线程池并发
        if self.n_parallel <= 1:
            return self.xtb_runner.run_batch(molecules, work_dir_base=work_dir)

        results = []
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            futures = {}
            for i, mol in enumerate(molecules):
                wdir = None
                if work_dir is not None:
                    wdir = f"{work_dir}/mol_{i:04d}"
                future = executor.submit(self.xtb_runner.run, mol, wdir)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    updated_mol = future.result()
                    results.append((idx, updated_mol))
                except Exception as e:
                    logger.error(f"XTB 并行异常 (mol {idx}): {e}")
                    mol = molecules[idx]
                    mol.xtb_error = f"并行异常: {e}"
                    results.append((idx, mol))

        # 恢复原始顺序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _run_pyscf_parallel(self, molecules: List[MoleculeData]) -> List[MoleculeData]:
        """并行运行 pySCF DFT 计算."""
        if self.n_parallel <= 1 or len(molecules) <= 1:
            return self.pyscf_runner.run_batch(molecules)

        results = []
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            futures = {}
            for i, mol in enumerate(molecules):
                future = executor.submit(self.pyscf_runner.run, mol)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    updated_mol = future.result()
                    results.append((idx, updated_mol))
                except Exception as e:
                    logger.error(f"pySCF 并行异常 (mol {idx}): {e}")
                    mol = molecules[idx]
                    mol.dft_error = f"并行异常: {e}"
                    results.append((idx, mol))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _build_output(
        self, molecules: List[MoleculeData], target_gap: float
    ) -> pd.DataFrame:
        """构建最终输出的结构化表格."""
        rows = []
        for mol in molecules:
            row = mol.to_dict()
            row["target_gap_eV"] = target_gap
            if mol.dft_gap is not None:
                row["dft_deviation_eV"] = round(abs(mol.dft_gap - target_gap), 4)
            else:
                row["dft_deviation_eV"] = None
            rows.append(row)

        df = pd.DataFrame(rows)

        # 按 DFT gap 偏差排序
        if "dft_deviation_eV" in df.columns:
            df = df.sort_values("dft_deviation_eV", ascending=True, na_position="last")

        return df

    def _ensure_correlation_model(self) -> None:
        """延迟加载或训练相关性校准模型."""
        if self._correlation_model is not None:
            return
        from .correlation import CorrelationModel
        logger.info("加载/训练 XTB→DFT 相关性校准模型...")
        self._correlation_model = CorrelationModel.load_or_train(
            n_samples=self._correlation_n_samples,
            force_retrain=self._correlation_force_retrain,
        )
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """从输出 DataFrame 生成汇总统计."""
        summary = {
            "n_total": len(df),
            "n_xtb_success": int(df["xtb_success"].sum()),
            "n_dft_success": int(df["dft_success"].sum()),
        }
        if "dft_gap_eV" in df.columns:
            valid = df[df["dft_gap_eV"].notna()]
            if len(valid) > 0:
                summary["dft_gap_mean_eV"] = round(float(valid["dft_gap_eV"].mean()), 4)
                summary["dft_gap_min_eV"] = round(float(valid["dft_gap_eV"].min()), 4)
                summary["dft_gap_max_eV"] = round(float(valid["dft_gap_eV"].max()), 4)
                summary["dft_deviation_min_eV"] = round(float(valid["dft_deviation_eV"].min()), 4)
        return summary
