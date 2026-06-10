"""SimpleEvaluator — 纯 XTB 分子评估器，无 DFT 计算.

流程:
    1. 接收 n 个 SMILES
    2. 并行运行 XTB 几何优化 (含 --dipole)
    3. 按 |xtb_gap - target_gap| 排序
    4. 返回前 m 个分子的 XTB 计算结果
"""

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_XTB_BIN = os.environ.get("XTB_BIN", "/home/jgong/calculations/xtb-dist/bin/xtb")
_XTB_OMP_NUM_THREADS = int(os.environ.get("XTB_OMP_NUM_THREADS", "4"))


# ── 偶极矩解析 ─────────────────────────────────────────────
def _parse_dipole(stdout: str) -> Optional[float]:
    """从 xtb 输出中提取总偶极矩 (Debye).

    格式 (需 --dipole 参数):
      molecular dipole:
                     x           y           z       tot (Debye)
       q only:       ...
         full:       -0.473       0.301      -0.549       1.994
    """
    for line in stdout.split("\n"):
        if line.strip().startswith("full:"):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    return float(parts[4])
                except ValueError:
                    pass
    return None


# ── 单分子结果 ─────────────────────────────────────────────
@dataclass
class SimpleMolResult:
    """单个分子的 XTB 计算结果."""

    smiles: str
    xtb_gap_eV: Optional[float] = None
    xtb_homo_eV: Optional[float] = None
    xtb_lumo_eV: Optional[float] = None
    xtb_dipole_D: Optional[float] = None
    xtb_energy_Ha: Optional[float] = None
    xtb_deviation: Optional[float] = None
    xtb_success: bool = False
    xtb_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "xtb_gap_eV": round(self.xtb_gap_eV, 4) if self.xtb_gap_eV is not None else None,
            "xtb_homo_eV": round(self.xtb_homo_eV, 4) if self.xtb_homo_eV is not None else None,
            "xtb_lumo_eV": round(self.xtb_lumo_eV, 4) if self.xtb_lumo_eV is not None else None,
            "xtb_dipole_D": round(self.xtb_dipole_D, 4) if self.xtb_dipole_D is not None else None,
            "xtb_energy_Ha": round(self.xtb_energy_Ha, 6) if self.xtb_energy_Ha is not None else None,
            "xtb_deviation_eV": round(self.xtb_deviation, 4) if self.xtb_deviation is not None else None,
            "xtb_success": self.xtb_success,
        }


# ── SMILES → XYZ ────────────────────────────────────────────
def _smiles_to_xyz_block(smiles: str, seed: int = 42) -> Optional[str]:
    """将 SMILES 转为 XYZ 格式字符串."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
        if result != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), ""]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s}  {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}")
    return "\n".join(lines)


# ── XTB 输出解析 ────────────────────────────────────────────
_GAP_PATTERN = re.compile(r"\|\s*HOMO-LUMO\s+GAP\s+([\d.]+)\s+eV\s*\|")
_ENERGY_PATTERN = re.compile(r"\|\s*TOTAL\s+ENERGY\s+([\-\d.Ee]+)\s+Eh\s*\|")


def _parse_xtb_output(stdout: str) -> dict:
    """从 xtb 标准输出中提取 gap, 能量, HOMO/LUMO, 偶极矩."""
    result: dict = {}

    m = _GAP_PATTERN.search(stdout)
    if m:
        result["gap_eV"] = float(m.group(1))
    m = _ENERGY_PATTERN.search(stdout)
    if m:
        result["total_energy_Eh"] = float(m.group(1))

    # HOMO / LUMO 能量 (eV)
    # 行格式: "  10        2.0000           -0.4112369             -11.1903 (HOMO)"
    lines = stdout.split("\n")
    last_homo_idx = last_lumo_idx = -1
    for i, line in enumerate(lines):
        if "(HOMO)" in line:
            last_homo_idx = i
        if "(LUMO)" in line:
            last_lumo_idx = i
    try:
        if last_homo_idx >= 0:
            parts = lines[last_homo_idx].split()
            idx = parts.index("(HOMO)")
            result["homo_eV"] = float(parts[idx - 1])
        if last_lumo_idx >= 0:
            parts = lines[last_lumo_idx].split()
            idx = parts.index("(LUMO)")
            result["lumo_eV"] = float(parts[idx - 1])
    except (ValueError, IndexError):
        pass

    dipole = _parse_dipole(stdout)
    if dipole is not None:
        result["dipole_D"] = dipole

    return result


# ── 单分子 XTB 运行 ─────────────────────────────────────────
def _run_xtb_single(smiles: str, xtb_bin: str, gfn: int, omp_threads: int,
                    timeout: int) -> SimpleMolResult:
    """对一个 SMILES 运行 XTB 几何优化 (含 --dipole) 并解析结果."""
    mol_result = SimpleMolResult(smiles=smiles)

    try:
        xyz_block = _smiles_to_xyz_block(smiles)
        if xyz_block is None:
            mol_result.xtb_error = f"无法生成 3D 构象: {smiles}"
            return mol_result

        with tempfile.TemporaryDirectory(prefix="xtb_") as tmpdir:
            xyz_path = os.path.join(tmpdir, "input.xyz")
            with open(xyz_path, "w") as f:
                f.write(xyz_block)

            env = {**os.environ, "OMP_NUM_THREADS": str(omp_threads)}
            cmd = [xtb_bin, xyz_path, "--gfn", str(gfn), "--opt", "--dipole"]

            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir, env=env,
            )

            if proc.returncode != 0:
                mol_result.xtb_error = f"xtb 退出码 {proc.returncode}: {proc.stderr[-300:]}"
                return mol_result

            parsed = _parse_xtb_output(proc.stdout)
            mol_result.xtb_gap_eV = parsed.get("gap_eV")
            mol_result.xtb_energy_Ha = parsed.get("total_energy_Eh")
            mol_result.xtb_homo_eV = parsed.get("homo_eV")
            mol_result.xtb_lumo_eV = parsed.get("lumo_eV")
            mol_result.xtb_dipole_D = parsed.get("dipole_D")

            if mol_result.xtb_gap_eV is None:
                mol_result.xtb_error = "无法解析 HOMO-LUMO gap"
                return mol_result

            mol_result.xtb_success = True

    except subprocess.TimeoutExpired:
        mol_result.xtb_error = f"XTB 超时 ({timeout}s)"
    except FileNotFoundError:
        mol_result.xtb_error = f"xtb 二进制未找到: {xtb_bin}"
    except Exception as e:
        mol_result.xtb_error = f"异常: {type(e).__name__}: {e}"

    return mol_result


# ── 主编排器 ────────────────────────────────────────────────
class SimpleEvaluator:
    """纯 XTB 分子评估器，无 DFT.

    用法:
        evaluator = SimpleEvaluator()
        df = evaluator.evaluate(
            smiles_list=["CCO", "CCN"],
            target_gap=3.0,
            n_top=2,
        )
        print(df)
    """

    def __init__(
        self,
        xtb_bin: Optional[str] = None,
        gfn: int = 2,
        omp_num_threads: Optional[int] = None,
        xtb_timeout: int = 300,
        n_parallel: int = 4,
    ):
        self.xtb_bin = xtb_bin or _XTB_BIN
        self.gfn = gfn
        self.omp_num_threads = omp_num_threads or _XTB_OMP_NUM_THREADS
        self.xtb_timeout = xtb_timeout
        self.n_parallel = n_parallel

    def evaluate(
        self,
        smiles_list: List[str],
        target_gap: float,
        n_top: int,
    ) -> pd.DataFrame:
        """执行 XTB 评估.

        Args:
            smiles_list: 候选 SMILES 列表.
            target_gap: 目标 HOMO-LUMO gap (eV).
            n_top: 返回的 top 分子数.

        Returns:
            pandas DataFrame，包含前 n_top 个分子的 XTB 计算结果.
        """
        n_input = len(smiles_list)
        logger.info(f"SimpleEvaluator: {n_input} SMILES, target_gap={target_gap}eV, n_top={n_top}")

        results: List[SimpleMolResult] = []

        if self.n_parallel <= 1:
            for smi in smiles_list:
                results.append(_run_xtb_single(
                    smi, self.xtb_bin, self.gfn,
                    self.omp_num_threads, self.xtb_timeout,
                ))
        else:
            with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                future_map = {}
                for smi in smiles_list:
                    fut = executor.submit(
                        _run_xtb_single, smi, self.xtb_bin, self.gfn,
                        self.omp_num_threads, self.xtb_timeout,
                    )
                    future_map[fut] = smi
                for fut in as_completed(future_map):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        smi = future_map[fut]
                        r = SimpleMolResult(smiles=smi, xtb_error=f"并行异常: {e}")
                        results.append(r)

        n_success = sum(1 for r in results if r.xtb_success)
        n_failed = n_input - n_success
        if n_failed > 0:
            logger.warning(f"XTB: {n_success} 成功, {n_failed} 失败")
        if n_success == 0:
            raise RuntimeError("所有分子的 XTB 计算均失败")

        for r in results:
            if r.xtb_success and r.xtb_gap_eV is not None:
                r.xtb_deviation = abs(r.xtb_gap_eV - target_gap)

        results.sort(key=lambda r: r.xtb_deviation if r.xtb_deviation is not None else float("inf"))
        top = results[:n_top]
        top = [r for r in top if r.xtb_success]

        if not top:
            raise RuntimeError("没有成功完成 XTB 的分子")

        df = pd.DataFrame([r.to_dict() for r in top])
        df["target_gap_eV"] = target_gap

        logger.info(f"返回 top {len(top)} 分子, gap 范围: "
                     f"[{top[0].xtb_gap_eV:.4f} .. {top[-1].xtb_gap_eV:.4f}] eV")
        return df
