"""分子指纹生成 — Morgan/ECFP4 指纹 + SMILES 转换."""

from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

_ATOMIC_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 18: "Ar", 35: "Br",
}

FP_RADIUS = 2
FP_BITS = 2048


def xyz_to_fingerprint(z: np.ndarray, pos: np.ndarray) -> Optional[np.ndarray]:
    """从 QM9 的原子序数和坐标生成 Morgan 指纹 (2048-bit 二值向量).

    Returns:
        (FP_BITS,) 的 uint8 数组 (0/1)，或 None 如果分子无法解析.
    """
    mol = _xyz_to_mol(z, pos)
    if mol is None:
        return None
    return smiles_to_fingerprint(Chem.MolToSmiles(mol))


def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """从 SMILES 生成 Morgan 指纹.

    Returns:
        (FP_BITS,) 的 uint8 数组 (0/1)，或 None 如果 SMILES 无效.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    for i in range(FP_BITS):
        if fp.GetBit(i):
            arr[i] = 1
    return arr


def xyz_to_smiles(z: np.ndarray, pos: np.ndarray) -> Optional[str]:
    """QM9 XYZ → SMILES."""
    mol = _xyz_to_mol(z, pos)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _xyz_to_mol(z: np.ndarray, pos: np.ndarray):
    """从原子序数和坐标构建 RDKit 分子对象（带键感知）."""
    try:
        lines = [str(len(z)), ""]
        for atom_z, coord in zip(z, pos):
            sym = _ATOMIC_SYMBOLS.get(int(atom_z), "X")
            lines.append(f"{sym:2s}  {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
        mol = Chem.MolFromXYZBlock("\n".join(lines))
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None
