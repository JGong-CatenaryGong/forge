"""MolAgent Evaluator — 分子属性评估系统.

两级计算流程:
  XTB (GFN2-xTB) 几何优化 + 预筛选
  → [可选: XTB→DFT 相关性校准]
  → pySCF (DFT) 精确属性计算
  → 结构化输出

主要入口:
    from evaluator import Evaluator
    evaluator = Evaluator(use_correlation=True)
    df = evaluator.evaluate(smiles_list, target_gap=5.0, n_top=3)
"""

from .evaluator import Evaluator
from .molecule import MoleculeData
from .xtb_runner import XTBRunner
from .pyscf_runner import PySCFRunner
from .correlation import CorrelationModel

__all__ = ["Evaluator", "MoleculeData", "XTBRunner", "PySCFRunner", "CorrelationModel"]
