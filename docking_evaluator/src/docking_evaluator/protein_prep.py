"""蛋白质准备 — PDBFixer 修复 + 加氢 → PDB 文件."""

import logging, os, tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def prepare_protein(pdb_path: str, output_dir: Optional[str] = None,
                    add_hs: bool = True, ph: float = 7.4) -> Optional[str]:
    """用 PDBFixer 准备蛋白质, 返回修复后的 PDB 文件路径."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        logger.error("pdbfixer/openmm 未安装: pip install pdbfixer")
        return None

    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if add_hs:
        fixer.addMissingHydrogens(ph)

    out_dir = output_dir or os.path.dirname(pdb_path)
    out_path = os.path.join(out_dir, "receptor_prepared.pdb")
    with open(out_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    logger.info(f"蛋白质已准备: {out_path}")
    return out_path
