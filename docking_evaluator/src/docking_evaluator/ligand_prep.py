"""配体准备 — SMILES → Meeko → PDBQT."""

import logging, os, tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def smiles_to_pdbqt(smiles: str, output_path: Optional[str] = None, ph: float = 7.4) -> Optional[str]:
    """SMILES → RDKit 3D → Meeko → PDBQT (含 Gasteiger 电荷和 AD 类型).

    Args:
        smiles: 输入 SMILES.
        output_path: 输出 PDBQT 文件路径 (可选).
        ph: pH 值.

    Returns:
        PDBQT 字符串, 或 None.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"无效 SMILES: {smiles}")
        return None
    mol = Chem.AddHs(mol)

    # 3D 构象生成
    from rdkit.Chem import AllChem
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        logger.warning(f"无法生成 3D 构象: {smiles}")
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    # 用 Meeko 写 PDBQT (含 Gasteiger 电荷 + AD4 原子类型)
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        preparator = MoleculePreparation()
        mol_setup = preparator.prepare(mol)[0]  # 取第一个互变异构/质子化状态
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup)
        if not is_ok:
            logger.warning(f"Meeko PDBQT 写出失败: {error_msg}")
            return None
    except ImportError:
        logger.warning("Meeko 未安装, 回退到手动 PDBQT. pip install meeko")
        pdbqt_string = _manual_pdbqt(mol)
        if pdbqt_string is None:
            return None

    if output_path:
        with open(output_path, "w") as f:
            f.write(pdbqt_string)
    return pdbqt_string


def _manual_pdbqt(mol) -> Optional[str]:
    """回退: RDKit Gasteiger 电荷 + 手动 PDBQT."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None

    _AD_TYPE = {6: "C", 7: "NA", 8: "OA", 9: "F", 15: "P", 16: "SA", 17: "Cl", 35: "Br", 53: "I"}
    lines = []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        anum = atom.GetAtomicNum()
        charge = float(atom.GetPropsAsDict().get("_GasteigerCharge", 0.0))
        ad_type = _AD_TYPE.get(anum, "A")
        lines.append(
            f"HETATM{i+1:5d} {atom.GetSymbol():2s}  LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:6.3f} {ad_type:2s}"
        )
    return "\n".join(lines) + "\nEND\n"
