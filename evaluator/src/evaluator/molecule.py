"""分子数据结构 — 承载SMILES→XTB→pySCF全流程的属性数据."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class MoleculeData:
    """一个分子在整个评估流程中的状态和计算结果.

    字段按计算阶段分组:
    - 输入: smiles
    - XTB阶段: xtb_gap, xtb_energy, atomic_numbers, positions
    - pySCF阶段: dft_gap, dft_homo, dft_lumo, dft_dipole, dft_mulliken_charge, dft_energy
    - 状态标记: xtb_success, dft_success, xtb_error, dft_error
    """

    smiles: str

    # ---- XTB 阶段 ----
    xtb_gap: Optional[float] = None          # HOMO-LUMO gap, eV
    xtb_energy: Optional[float] = None       # 总能量, Hartree
    xtb_homo: Optional[float] = None         # HOMO 能量, eV
    xtb_lumo: Optional[float] = None         # LUMO 能量, eV
    atomic_numbers: Optional[List[int]] = None   # 原子序数列表
    positions: Optional[np.ndarray] = None       # 优化后的坐标 (n_atoms, 3), Å

    # ---- 相关性校准阶段 ----
    calib_homo: Optional[float] = None       # 校准后 HOMO, eV
    calib_lumo: Optional[float] = None       # 校准后 LUMO, eV
    calib_gap: Optional[float] = None        # 校准后 gap, eV

    # ---- pySCF DFT 阶段 ----
    dft_gap: Optional[float] = None          # HOMO-LUMO gap, eV
    dft_homo: Optional[float] = None         # HOMO 能量, eV
    dft_lumo: Optional[float] = None         # LUMO 能量, eV

    # ---- 状态 ----
    xtb_success: bool = False
    dft_success: bool = False
    xtb_error: Optional[str] = None
    dft_error: Optional[str] = None

    @property
    def xtb_deviation(self) -> Optional[float]:
        """XTB gap 与目标之间的偏差（绝对值），由 Evaluator 设置."""
        return getattr(self, "_xtb_deviation", None)

    @xtb_deviation.setter
    def xtb_deviation(self, value: Optional[float]) -> None:
        self._xtb_deviation = value

    @property
    def n_atoms(self) -> int:
        """原子数."""
        if self.atomic_numbers is not None:
            return len(self.atomic_numbers)
        if self.positions is not None:
            return len(self.positions)
        return 0

    def to_dict(self) -> dict:
        """转为可序列化的字典，用于输出表格."""
        return {
            "smiles": self.smiles,
            "xtb_gap_eV": round(self.xtb_gap, 4) if self.xtb_gap is not None else None,
            "xtb_homo_eV": round(self.xtb_homo, 4) if self.xtb_homo is not None else None,
            "xtb_lumo_eV": round(self.xtb_lumo, 4) if self.xtb_lumo is not None else None,
            "xtb_deviation_eV": round(self.xtb_deviation, 4) if self.xtb_deviation is not None else None,
            "calib_gap_eV": round(self.calib_gap, 4) if self.calib_gap is not None else None,
            "calib_homo_eV": round(self.calib_homo, 4) if self.calib_homo is not None else None,
            "calib_lumo_eV": round(self.calib_lumo, 4) if self.calib_lumo is not None else None,
            "dft_gap_eV": round(self.dft_gap, 4) if self.dft_gap is not None else None,
            "dft_homo_eV": round(self.dft_homo, 4) if self.dft_homo is not None else None,
            "dft_lumo_eV": round(self.dft_lumo, 4) if self.dft_lumo is not None else None,
            "dft_dipole_D": round(self.dft_dipole, 4) if self.dft_dipole is not None else None,
            "dft_energy_Ha": round(self.dft_energy, 6) if self.dft_energy is not None else None,
            "dft_mulliken_charge": (
                [round(c, 4) for c in self.dft_mulliken_charge]
                if self.dft_mulliken_charge is not None else None
            ),
            "xtb_success": self.xtb_success,
            "dft_success": self.dft_success,
        }