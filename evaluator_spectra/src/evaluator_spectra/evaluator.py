"""SpectraEvaluator — XTB → ORCA DFT → TD-DFT → 透射率 → CIE 1931."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .molecule import MoleculeSpectra
from .tddft_runner import TDDTFTRunner
from .spectrum import compute_spectrum

logger = logging.getLogger(__name__)


class SpectraEvaluator:
    def __init__(self, xtb_bin=None, xtb_gfn=2, xtb_omp_threads=4, xtb_timeout=300,
                 orca_bin="/home/jgong/calculations/orca/orca",
                 functional="PBE0", basis="def2-SVP", nstates=10, orca_timeout=600,
                 n_parallel=2, spectrum_fwhm=0.3, spectrum_n_points=1000,
                 spectrum_concentration=1.0):
        self.runner = TDDTFTRunner(xtb_bin=xtb_bin, xtb_gfn=xtb_gfn, xtb_omp_threads=xtb_omp_threads,
                                   xtb_timeout=xtb_timeout, orca_bin=orca_bin,
                                   functional=functional, basis=basis,
                                   nstates=nstates, orca_timeout=orca_timeout)
        self.n_parallel = n_parallel
        self.spectrum_fwhm = spectrum_fwhm
        self.spectrum_n_points = spectrum_n_points
        self.spectrum_concentration = spectrum_concentration

    def evaluate(self, smiles_list, target_cie, n_top):
        """完整评估: XTB → ORCA → 透射率 → CIE 1931，按感知色偏差排序."""
        tx, ty = target_cie
        logger.info(f"SpectraEvaluator: {len(smiles_list)} SMILES, target CIE=({tx:.4f},{ty:.4f}), n_top={n_top}")

        molecules = []
        def _run(smi):
            m = self.runner.run(smi)
            if m.td_dft_success:
                compute_spectrum(m, fwhm=self.spectrum_fwhm, n_points=self.spectrum_n_points,
                                 concentration=self.spectrum_concentration)
            return m

        if self.n_parallel <= 1:
            for smi in smiles_list:
                molecules.append(_run(smi))
        else:
            with ThreadPoolExecutor(max_workers=self.n_parallel) as ex:
                fs = {ex.submit(_run, smi): smi for smi in smiles_list}
                for f in as_completed(fs):
                    try:
                        molecules.append(f.result())
                    except Exception as e:
                        molecules.append(MoleculeSpectra(smiles=fs[f], error=f"并行异常: {e}"))

        n_ok = sum(1 for m in molecules if m.td_dft_success)
        logger.info(f"ORCA TD-DFT: {n_ok}/{len(molecules)} OK")

        # 按 CIE 偏差排序
        for m in molecules:
            dev = np.sqrt((m.cie_x - tx)**2 + (m.cie_y - ty)**2) if m.cie_x is not None else float("inf")
            m._cie_deviation = dev
        molecules.sort(key=lambda m: getattr(m, "_cie_deviation", float("inf")))
        top = [m for m in molecules[:n_top] if m.td_dft_success]
        if not top:
            raise RuntimeError("没有成功完成 TD-DFT 的分子")

        records = []
        for m in top:
            rec = m.to_dict()
            rec["cie_deviation"] = round(m._cie_deviation, 6) if m._cie_deviation != float("inf") else None
            rec["target_cie_x"] = tx
            rec["target_cie_y"] = ty
            records.append(rec)

        df = pd.DataFrame(records)
        logger.info(f"返回 top {len(top)} CIE: [({top[0].cie_x:.3f},{top[0].cie_y:.3f}) .. ({top[-1].cie_x:.3f},{top[-1].cie_y:.3f})]")
        return df
