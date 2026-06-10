"""pySCF DFT 接口 — 对 XTB 优化后的几何构型进行精确 DFT 计算."""

import os
from typing import List, Optional, Tuple

import numpy as np

from .molecule import MoleculeData


# Hartree → eV
_HA_TO_EV = 27.211386245988

# 元素序数 → 符号（缓存）
_ATOMIC_NUMBERS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo",
    43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce",
    59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy",
    67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
    83: "Bi", 84: "Po", 85: "At", 86: "Rn",
}


def _build_pyscf_atom_str(atomic_numbers: List[int], positions: np.ndarray) -> str:
    """构建 pySCF 的 atom 字符串.

    格式: "C  0.000  0.000  0.000; H  1.000  0.000  0.000"
    """
    parts = []
    for z, pos in zip(atomic_numbers, positions):
        symbol = _ATOMIC_NUMBERS.get(z, f"Z{z}")
        parts.append(f"{symbol:2s}  {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}")
    return "; ".join(parts)


class PySCFRunner:
    """使用 pySCF 进行 DFT 单点能计算.

    用法:
        runner = PySCFRunner(functional="B3LYP", basis="6-31G(2d,p)")
        mol = runner.run(mol)  # mol 必须有 atomic_numbers 和 positions
    """

    def __init__(
        self,
        functional: str = "B3LYP",
        basis: str = "6-31G(2d,p)",
        charge: int = 0,
        spin: int = 0,
        max_scf_cycles: int = 200,
        verbose: int = 0,
    ):
        """
        Args:
            functional: DFT 泛函名称 (XC), 支持 pySCF 内置泛函和自定义.
            basis: 基组名称.
            charge: 分子总电荷.
            spin: 自旋多重度 (2S+1, 闭壳层=0 表示单重态).
            max_scf_cycles: SCF 最大迭代步数.
            verbose: pySCF 输出详细程度 (0=静默).
        """
        self.functional = functional
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.max_scf_cycles = max_scf_cycles
        self.verbose = verbose

    def run(self, mol: MoleculeData) -> MoleculeData:
        """对分子进行 DFT 单点能计算.

        要求 mol 必须有 atomic_numbers 和 positions（即已完成 XTB 优化）.

        Args:
            mol: 包含优化后几何结构的 MoleculeData.

        Returns:
            更新了 dft_gap, dft_homo, dft_lumo, dft_dipole, dft_mulliken_charge, dft_energy 的 MoleculeData.
        """
        if mol.atomic_numbers is None or mol.positions is None:
            mol.dft_error = "pySCF 需要原子序数和坐标（请先完成 XTB 优化）"
            return mol

        try:
            from pyscf import gto, dft

            # 1. 构建分子
            atom_str = _build_pyscf_atom_str(mol.atomic_numbers, mol.positions)
            pymol = gto.M(
                atom=atom_str,
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                verbose=self.verbose,
            )
            pymol.build()

            # 2. DFT 计算
            mf = dft.RKS(pymol) if self.spin == 0 else dft.UKS(pymol)
            mf.xc = self.functional
            mf.max_cycle = self.max_scf_cycles
            mf.verbose = 0
            mf.kernel()

            if not mf.converged:
                mol.dft_error = "pySCF SCF 未收敛"
                return mol

            # 3. 提取 HOMO/LUMO
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ

            homo_eV, lumo_eV = _get_homo_lumo(mo_energy, mo_occ)
            mol.dft_homo = homo_eV
            mol.dft_lumo = lumo_eV
            if homo_eV is not None and lumo_eV is not None:
                mol.dft_gap = lumo_eV - homo_eV

            # 4. 偶极矩
            dipole = mf.dip_moment(verbose=0)
            if dipole is not None:
                mol.dft_dipole_vec = np.array(dipole)
                mol.dft_dipole = float(np.linalg.norm(dipole))

            # 5. Mulliken 电荷
            mol.dft_mulliken_charge = _get_mulliken_charges(mf, pymol)

            # 6. 总能量 (Hartree)
            mol.dft_energy = float(mf.e_tot)

            mol.dft_success = True

        except Exception as e:
            mol.dft_error = f"pySCF 运行时异常: {type(e).__name__}: {e}"

        return mol


    def run_batch(self, molecules: List[MoleculeData]) -> List[MoleculeData]:
        """批量运行 DFT 计算（串行）."""
        return [self.run(mol) for mol in molecules]


def _get_homo_lumo(
    mo_energy: np.ndarray, mo_occ: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    """从轨道能量和占据数中提取 HOMO / LUMO 能量 (eV).

    Args:
        mo_energy: 轨道能量数组 (Hartree).
        mo_occ: 轨道占据数数组.

    Returns:
        (homo_eV, lumo_eV).
    """
    # 占据数 > 0 的是占据轨道，= 0 的是虚轨道
    occupied_mask = mo_occ > 0
    virtual_mask = ~occupied_mask

    homo_eV = None
    lumo_eV = None

    if np.any(occupied_mask):
        homo_eV = float(np.max(mo_energy[occupied_mask])) * _HA_TO_EV
    if np.any(virtual_mask):
        lumo_eV = float(np.min(mo_energy[virtual_mask])) * _HA_TO_EV

    return homo_eV, lumo_eV


def _get_mulliken_charges(mf, mol) -> Optional[List[float]]:
    """提取 Mulliken 电荷分布."""
    try:
        from pyscf.scf.hf import mulliken_pop
        dm = mf.make_rdm1()
        _, charges = mulliken_pop(mol, dm, verbose=0)
        return [float(c) for c in charges]
    except Exception:
        return None
