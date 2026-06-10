"""工具函数：XYZ文件读写、XTB输出解析."""

import re
import os
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# ---- XYZ 文件 ----

def smiles_to_xyz_block(smiles: str, seed: int = 42) -> Optional[str]:
    """将 SMILES 转为 XYZ 格式字符串（包含原子坐标）.

    Args:
        smiles: SMILES 字符串.
        seed: RDKit 构象生成的随机种子.

    Returns:
        XYZ 格式的多行字符串，或 None（转换失败时）.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    # 使用 ETKDG 方法生成 3D 构象
    result = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if result != 0:
        # 回退到基本距离几何
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
        if result != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # MMFF 可能对某些原子类型失败，用 UFF 回退
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass  # 即使力场优化失败，仍可使用初始构象

    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), ""]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s}  {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}")
    return "\n".join(lines)


def write_xyz(xyz_block: str, path: str) -> None:
    """将 XYZ 块写入文件."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(xyz_block)


def read_xyz(path: str) -> Tuple[List[int], np.ndarray]:
    """从 XYZ 文件读取原子序数和坐标.

    Returns:
        (atomic_numbers, positions): 原子序数列表和 (n_atoms, 3) 坐标数组.
    """
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    atomic_numbers = []
    positions = np.zeros((n_atoms, 3))
    # 第2行是注释行（能量信息），从第3行开始是原子坐标
    for i in range(n_atoms):
        parts = lines[i + 2].split()
        symbol = parts[0]
        # 处理元素符号（去除可能的数字后缀）
        atomic_numbers.append(_symbol_to_number(symbol))
        positions[i] = [float(x) for x in parts[1:4]]
    return atomic_numbers, positions


def _symbol_to_number(symbol: str) -> int:
    """元素符号 → 原子序数."""
    # 去除可能的同位素标记等
    symbol = re.sub(r"[^a-zA-Z]", "", symbol)
    from rdkit.Chem import GetPeriodicTable
    return GetPeriodicTable().GetAtomicNumber(symbol)


# ---- XTB 输出解析 ----

_GAP_PATTERN = re.compile(r"HOMO-LUMO\s+GAP\s+([\d.]+)\s+eV")
_TOTAL_ENERGY_PATTERN = re.compile(r"TOTAL\s+ENERGY\s+([\-\d.]+)\s+Eh")
_HL_GAP_PATTERN = re.compile(r"HL-Gap\s+[\d.]+\s+Eh\s+([\d.]+)\s+eV")
_HOMO_PATTERN = re.compile(r"\(HOMO\)\s*$")


def parse_xtb_output(stdout: str) -> dict:
    """从 xtb 标准输出中提取关键属性.

    Returns:
        dict with keys: gap_eV, total_energy_Eh, homo_eV, lumo_eV.
    """
    result: dict = {}

    # 优先使用末尾的 "| ... |" 格式（最终汇总）
    gap_match = _GAP_PATTERN.search(stdout)
    energy_match = _TOTAL_ENERGY_PATTERN.search(stdout)

    if gap_match:
        result["gap_eV"] = float(gap_match.group(1))
    if energy_match:
        result["total_energy_Eh"] = float(energy_match.group(1))

    # 也尝试 HL-Gap 行作为备选
    if "gap_eV" not in result:
        hl_match = _HL_GAP_PATTERN.findall(stdout)
        if hl_match:
            result["gap_eV"] = float(hl_match[-1])

    return result


def parse_xtb_orbital_energies(stdout: str) -> Optional[Tuple[float, float]]:
    """从 xtb 标准输出中提取 HOMO 和 LUMO 能量 (eV).

    解析格式: "      10   2.0000   -0.4118   -11.2057 (HOMO)"
              "      11            -0.0123    -0.3345 (LUMO)"

    Returns:
        (homo_eV, lumo_eV) 或 None.
    """
    lines = stdout.split("\n")

    # 找最后的 (HOMO) 和 (LUMO) 行
    last_homo_idx = -1
    last_lumo_idx = -1
    for i, line in enumerate(lines):
        if "(HOMO)" in line:
            last_homo_idx = i
        if "(LUMO)" in line:
            last_lumo_idx = i
    try:
        # HOMO: eV 在 (HOMO) 标记前一列
        homo_parts = lines[last_homo_idx].split()
        homo_marker = homo_parts.index("(HOMO)")
        homo_eV = float(homo_parts[homo_marker - 1])

        # LUMO: eV 在 (LUMO) 标记前一列
        lumo_parts = lines[last_lumo_idx].split()
        lumo_marker = lumo_parts.index("(LUMO)")
        lumo_eV = float(lumo_parts[lumo_marker - 1])

        return homo_eV, lumo_eV
    except (ValueError, IndexError):
        return None