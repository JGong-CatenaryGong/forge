"""Smina 对接执行器."""

import logging
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple

from .molecule import DockingPose, DockingResult

logger = logging.getLogger(__name__)

_SCORE_REMARK = re.compile(r"REMARK\s+minimizedAffinity\s+(-?\d+\.\d+)")


def run_smina(
    ligand_pdbqt: str,
    receptor_pdb: str,
    smina_bin: str = "smina",
    scoring: str = "vinardo",
    num_modes: int = 9,
    exhaustiveness: int = 8,
    center: Optional[Tuple[float, float, float]] = None,
    size: Optional[Tuple[float, float, float]] = None,
) -> DockingResult:
    """运行 Smina 对接.

    Args:
        ligand_pdbqt: 配体 PDBQT 字符串.
        receptor_pdb: 受体 PDB 文件路径.
        smina_bin: Smina 二进制路径.
        scoring: 打分函数 (vinardo, vina, ad4).
        num_modes: 输出构象数.
        exhaustiveness: 搜索彻底度.
        center: 对接盒子中心 (x, y, z).
        size: 对接盒子大小 (x, y, z).

    Returns:
        DockingResult 对象.
    """
    result = DockingResult(smiles="")

    # 写临时文件
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w") as lf:
        lf.write(ligand_pdbqt)
        lig_path = lf.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as of:
        out_path = of.name

    cmd = [
        smina_bin,
        "--receptor", receptor_pdb,
        "--ligand", lig_path,
        "--out", out_path,
        "--scoring", scoring,
        "--num_modes", str(num_modes),
        "--exhaustiveness", str(exhaustiveness),
    ]
    if center is not None and size is not None:
        cmd += [
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
        ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        result.error = "Smina 超时 (300s)"
        _cleanup(lig_path, out_path)
        return result
    except FileNotFoundError:
        result.error = f"Smina 未找到: {smina_bin}"
        _cleanup(lig_path, out_path)
        return result

    _cleanup(lig_path)  # keep out_path for parsing

    if proc.returncode != 0:
        result.error = f"Smina 退出码 {proc.returncode}: {proc.stderr[-300:]}"
        _cleanup(out_path)
        return result

    # 解析输出: 构象分数
    if not os.path.exists(out_path):
        result.error = "Smina 未生成输出文件"
        return result
    with open(out_path) as f:
        pdbqt = f.read()
    os.unlink(out_path)

    poses = _parse_smina_output(pdbqt)
    if not poses:
        result.error = "未能解析对接构象"
        return result

    result.best_pose = poses[0]
    result.best_score = poses[0].score
    result.all_scores = [p.score for p in poses]
    result.docking_success = True
    return result


def _parse_smina_output(pdbqt_text: str) -> List[DockingPose]:
    """从 Smina 输出的 PDBQT 中提取对接构象和分数."""
    blocks = pdbqt_text.split("ENDMDL")
    poses = []
    for i, block in enumerate(blocks[:-1]):  # 最后一个 ENDMDL 后的内容忽略
        scores = _SCORE_REMARK.findall(block)
        if scores:
            poses.append(DockingPose(
                rank=i + 1,
                score=float(scores[0]),
                pdbqt_block=block.strip() + "\nENDMDL",
            ))
    return sorted(poses, key=lambda p: p.score)


def _cleanup(*paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass
