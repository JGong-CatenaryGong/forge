"""DockingEvaluator — 分子对接评估器.

接收 n 个 SMILES, 对特定蛋白质进行对接实验,
返回对接 score 和相互作用分析.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import pandas as pd

from .molecule import DockingResult
from .docking_runner import run_smina
from .interaction import analyze_interactions
from .ligand_prep import smiles_to_pdbqt
from .protein_prep import prepare_protein

logger = logging.getLogger(__name__)


class DockingEvaluator:
    """分子对接评估器.

    用法:
        evaluator = DockingEvaluator(receptor_pdb="protein.pdb", smina_bin="smina")
        df = evaluator.evaluate(
            smiles_list=["CCO", "c1ccccc1"],
            n_top=5,
        )
    """

    def __init__(
        self,
        receptor_pdb: str,
        smina_bin: str = "smina",
        scoring: str = "vinardo",
        num_modes: int = 9,
        exhaustiveness: int = 8,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        interaction_method: str = "prolif",
        n_parallel: int = 4,
        ph: float = 7.4,
    ):
        self.receptor_pdb = receptor_pdb
        self.smina_bin = smina_bin
        self.scoring = scoring
        self.num_modes = num_modes
        self.exhaustiveness = exhaustiveness
        self.center = center
        self.size = size
        self.interaction_method = interaction_method
        self.n_parallel = n_parallel
        self.ph = ph

        # 预先准备受体
        # 直接用原始 PDB (PDBFixer 会破坏分数)
        self._receptor_prepared = receptor_pdb
        # self._receptor_prepared = prepare_protein(receptor_pdb, add_hs=True, ph=ph)

    def evaluate(
        self,
        smiles_list: List[str],
        n_top: int = 5,
    ) -> pd.DataFrame:
        """执行批量对接评估.

        Args:
            smiles_list: 候选 SMILES 列表.
            n_top: 返回的 top 分子数 (按对接分数排序).

        Returns:
            DataFrame, 包含对接分数 + 相互作用分析.
        """
        logger.info(f"DockingEvaluator: {len(smiles_list)} SMILES, n_top={n_top}")

        results: List[DockingResult] = []

        def _run(smi):
            r = DockingResult(smiles=smi)
            pdbqt = smiles_to_pdbqt(smi, ph=self.ph)
            if pdbqt is None:
                r.error = "配体准备失败"
                return r

            r = run_smina(
                pdbqt, self._receptor_prepared,
                smina_bin=self.smina_bin, scoring=self.scoring,
                num_modes=self.num_modes, exhaustiveness=self.exhaustiveness,
                center=self.center, size=self.size,
            )
            r.smiles = smi

            if r.docking_success:
                if self.interaction_method:
                    r = analyze_interactions(r, self._receptor_prepared, method=self.interaction_method)
            return r

        if self.n_parallel <= 1:
            for smi in smiles_list:
                results.append(_run(smi))
        else:
            with ThreadPoolExecutor(max_workers=self.n_parallel) as ex:
                fs = {ex.submit(_run, s): s for s in smiles_list}
                for f in as_completed(fs):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        results.append(DockingResult(smiles=fs[f], error=f"异常: {e}"))

        n_ok = sum(1 for r in results if r.docking_success)
        logger.info(f"对接: {n_ok}/{len(results)} OK")

        # 按对接分数排序 (越低越好)
        results.sort(key=lambda r: r.best_score if r.best_score is not None else float("inf"))
        top = [r for r in results[:n_top] if r.docking_success]

        if not top:
            raise RuntimeError("没有成功完成对接的分子")

        df = pd.DataFrame([r.to_dict() for r in top])
        logger.info(f"返回 top {len(top)} 分子, 分数: [{top[0].best_score:.2f} .. {top[-1].best_score:.2f}]")
        return df
