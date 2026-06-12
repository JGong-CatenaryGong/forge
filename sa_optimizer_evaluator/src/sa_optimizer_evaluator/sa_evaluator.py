"""SA-Optimizer Evaluator — SA 评分优化 + ORCA S0-S1 激发能保持.

目标: 在 S0-S1 激发能不显著偏离目标值的前提下, 最小化 SA Score (合成更容易).
排序: |S1 - target| < tolerance → SA score 升序.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .sascorer import calc_sa_score

logger = logging.getLogger(__name__)


@dataclass
class MoleculeSA:
    """分子 SA-优化数据."""

    smiles: str
    sa_score: Optional[float] = None
    sa_valid: bool = False

    # ── ORCA TD-DFT ──
    s1_energy_eV: Optional[float] = None
    s1_wavelength_nm: Optional[float] = None
    s1_oscillator_strength: Optional[float] = None
    dft_gap_eV: Optional[float] = None
    dft_homo_eV: Optional[float] = None
    dft_lumo_eV: Optional[float] = None
    dft_dipole_D: Optional[float] = None
    dft_energy_Ha: Optional[float] = None
    orca_success: bool = False

    # ── Combined score ──
    s1_deviation: Optional[float] = None
    combined_rank: Optional[float] = None

    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "smiles": self.smiles,
            "sa_score": round(self.sa_score, 4) if self.sa_score is not None else None,
            "s1_energy_eV": round(self.s1_energy_eV, 4) if self.s1_energy_eV is not None else None,
            "s1_wavelength_nm": round(self.s1_wavelength_nm, 1) if self.s1_wavelength_nm is not None else None,
            "s1_oscillator_strength": round(self.s1_oscillator_strength, 6) if self.s1_oscillator_strength is not None else None,
            "dft_gap_eV": round(self.dft_gap_eV, 4) if self.dft_gap_eV is not None else None,
            "dft_homo_eV": round(self.dft_homo_eV, 4) if self.dft_homo_eV is not None else None,
            "dft_lumo_eV": round(self.dft_lumo_eV, 4) if self.dft_lumo_eV is not None else None,
            "dft_dipole_D": round(self.dft_dipole_D, 4) if self.dft_dipole_D is not None else None,
            "dft_energy_Ha": round(self.dft_energy_Ha, 6) if self.dft_energy_Ha is not None else None,
            "s1_deviation_eV": round(self.s1_deviation, 4) if self.s1_deviation is not None else None,
            "orca_success": self.orca_success,
        }
        return d


class SAOptimizerEvaluator:
    """SA-优化评估器: SA Score + ORCA TD-DFT S1 激发能.

    用法:
        evaluator = SAOptimizerEvaluator(n_parallel=3)
        df = evaluator.evaluate(
            smiles_list=["CCO", "c1ccccc1"],
            target_s1=2.5,      # 目标 S0-S1 激发能 (eV)
            s1_tolerance=0.3,   # 允许偏差 (eV)
            n_top=5,
        )
    """

    def __init__(
        self,
        n_parallel: int = 2,
        functional: str = "PBE0",
        basis: str = "def2-SVP",
        s1_tolerance: float = 0.3,
    ):
        from evaluator_spectra.tddft_runner import TDDTFTRunner

        self.runner = TDDTFTRunner(
            functional=functional, basis=basis, nstates=5,
            orca_timeout=600,
        )
        self.n_parallel = n_parallel
        self.s1_tolerance = s1_tolerance

    def evaluate(
        self,
        smiles_list: List[str],
        target_s1: float,
        s1_tolerance: Optional[float] = None,
        n_top: int = 5,
    ) -> pd.DataFrame:
        """执行 SA-优化评估.

        Args:
            smiles_list: 候选 SMILES.
            target_s1: 目标 S0-S1 激发能 (eV).
            s1_tolerance: 允许的 S1 偏差 (eV); 默认使用 __init__ 的值.
            n_top: 返回的 top 分子数.

        Returns:
            DataFrame, 按 S1 偏差 + SA score 综合排序.
        """
        tol = s1_tolerance if s1_tolerance is not None else self.s1_tolerance
        logger.info(f"SAOptimizer: {len(smiles_list)} SMILES, target S1={target_s1:.2f}eV, tol={tol:.2f}, n_top={n_top}")

        molecules: List[MoleculeSA] = []

        def _run(smi):
            mol = MoleculeSA(smiles=smi)
            # SA Score
            try:
                mol.sa_score = calc_sa_score(smi)
                mol.sa_valid = True
            except Exception as e:
                mol.error = f"SA score: {e}"
                return mol

            # ORCA TD-DFT
            try:
                orca_result = self.runner.run(smi)
            except Exception as e:
                mol.error = f"ORCA error: {e}"
                return mol

            if orca_result.td_dft_success and orca_result.excited_states:
                s1 = orca_result.excited_states[0]
                mol.s1_energy_eV = s1.energy_eV
                mol.s1_wavelength_nm = s1.wavelength_nm
                mol.s1_oscillator_strength = s1.oscillator_strength
                mol.orca_success = True

                mol.dft_gap_eV = orca_result.dft_gap_eV
                mol.dft_homo_eV = orca_result.dft_homo_eV
                mol.dft_lumo_eV = orca_result.dft_lumo_eV
                mol.dft_dipole_D = orca_result.dft_dipole_D
                mol.dft_energy_Ha = orca_result.dft_energy_Ha
            else:
                mol.error = orca_result.error or "ORCA failed"

            return mol

        if self.n_parallel <= 1:
            for smi in smiles_list:
                molecules.append(_run(smi))
        else:
            with ThreadPoolExecutor(max_workers=self.n_parallel) as ex:
                fs = {ex.submit(_run, s): s for s in smiles_list}
                for f in as_completed(fs):
                    try:
                        molecules.append(f.result())
                    except Exception as e:
                        molecules.append(MoleculeSA(smiles=fs[f], error=f"并行: {e}"))

        n_orca = sum(1 for m in molecules if m.orca_success)
        logger.info(f"ORCA: {n_orca}/{len(molecules)} OK")

        # ── 排序 ──
        for m in molecules:
            m.s1_deviation = abs(m.s1_energy_eV - target_s1) if m.s1_energy_eV is not None else None
            # Combined score: within tolerance → SA priority; otherwise → S1 deviation priority
            if m.orca_success and m.s1_deviation is not None and m.sa_score is not None:
                in_tol = m.s1_deviation <= tol
                # rank = (within_tol ? 0 : 1) * 1000 + (within_tol ? SA_score : s1_deviation * 100)
                m.combined_rank = (
                    (0 if in_tol else 1) * 1000.0
                    + (m.sa_score if in_tol else m.s1_deviation * 100.0)
                )
            else:
                m.combined_rank = float("inf")

        molecules.sort(key=lambda m: m.combined_rank if m.combined_rank is not None else float("inf"))
        top = [m for m in molecules[:n_top] if m.orca_success]

        if not top:
            raise RuntimeError("没有成功完成 ORCA TD-DFT 的分子")

        records = []
        for m in top:
            rec = m.to_dict()
            rec["target_s1_eV"] = target_s1
            rec["s1_tolerance_eV"] = tol
            records.append(rec)

        df = pd.DataFrame(records)
        logger.info(f"返回 top {len(top)} 分子, S1: [{top[0].s1_energy_eV:.3f}..{top[-1].s1_energy_eV:.3f}] eV, "
                     f"SA: [{top[0].sa_score:.2f}..{top[-1].sa_score:.2f}]")
        return df
