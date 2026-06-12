"""FORGE Spectra Evaluator — TD-DFT 光谱评估 + CIE 1931 色度坐标.

计算流程:
    SMILES → RDKit 3D → XTB 优化 → pySCF DFT (PBE0/def2-SVP)
    → TD-DFT (nstates=10) → Gaussian 展宽 → CIE 1931 xy

返回字段:
    - CIE 1931 色度坐标 (cie_x, cie_y)
    - DFT HOMO-LUMO gap, HOMO, LUMO
    - 10 个激发态的激发能和振子强度
    - 最大吸收波长
    - 偶极矩, Mulliken 电荷, 电子能量

用法:
    from evaluator_spectra import SpectraEvaluator

    evaluator = SpectraEvaluator(n_parallel=2)
    df = evaluator.evaluate(
        smiles_list=["CCO", "c1ccccc1"],
        target_cie=(0.3, 0.6),
        n_top=3,
    )
"""

from .evaluator import SpectraEvaluator
from .molecule import MoleculeSpectra, ExcitedState

__all__ = ["SpectraEvaluator", "MoleculeSpectra", "ExcitedState"]
