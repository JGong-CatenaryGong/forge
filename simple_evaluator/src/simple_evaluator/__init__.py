"""FORGE Simple Evaluator — 纯 XTB 分子评估，无 DFT.

接收 n 个 SMILES，用 XTB 几何优化后排序，
返回前 m 个分子的 XTB 计算结果作为反馈。

反馈字段:
    - HOMO-LUMO gap (eV)
    - HOMO energy (eV)
    - LUMO energy (eV)
    - 偶极矩 (Debye)
    - XTB 总能量 (Hartree)

用法:
    from simple_evaluator import SimpleEvaluator
    evaluator = SimpleEvaluator()
    results = evaluator.evaluate(
        smiles_list=["CCO", "CCN"],
        target_gap=3.0,
        n_top=2,
    )
"""

from .simple_evaluator import SimpleEvaluator

__all__ = ["SimpleEvaluator"]
