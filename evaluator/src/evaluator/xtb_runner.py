"""XTB (GFN2-xTB) 接口 — 通过子进程调用 xtb 二进制进行几何优化."""

import os
import subprocess
import tempfile
from typing import List, Optional

from .molecule import MoleculeData
from .utils import smiles_to_xyz_block, write_xyz, read_xyz, parse_xtb_output, parse_xtb_orbital_energies

# 可配置的 xtb 二进制路径
_XTB_BIN = os.environ.get("XTB_BIN", "/home/jgong/calculations/xtb-dist/bin/xtb")
_XTB_OMP_NUM_THREADS = int(os.environ.get("XTB_OMP_NUM_THREADS", "4"))


class XTBRunner:
    """通过子进程调用 xtb 二进制进行 GFN2-xTB 几何优化.

    用法:
        runner = XTBRunner()
        mol = runner.run(MoleculeData(smiles="CCO"))
        if mol.xtb_success:
            print(mol.xtb_gap)
    """

    def __init__(
        self,
        xtb_bin: Optional[str] = None,
        gfn: int = 2,
        omp_num_threads: Optional[int] = None,
        timeout: int = 300,
    ):
        """
        Args:
            xtb_bin: xtb 二进制路径，默认从 XTB_BIN 环境变量获取.
            gfn: GFN 方法版本 (0, 1, 2).
            omp_num_threads: OpenMP 线程数.
            timeout: 单个分子 xtb 计算超时(秒).
        """
        self.xtb_bin = xtb_bin or _XTB_BIN
        self.gfn = gfn
        self.omp_num_threads = omp_num_threads or _XTB_OMP_NUM_THREADS
        self.timeout = timeout

    def run(self, mol: MoleculeData, work_dir: Optional[str] = None) -> MoleculeData:
        """对一个分子运行 GFN2-xTB 几何优化.

        流程: SMILES → 3D构象 → XYZ → xtb --gfn N --opt → 解析输出.

        Args:
            mol: 包含 smiles 的 MoleculeData.
            work_dir: 工作目录, None 则使用临时目录.

        Returns:
            更新了 xtb_gap, xtb_energy, atomic_numbers, positions 等字段的 MoleculeData.
        """
        try:
            # 1. SMILES → XYZ
            xyz_block = smiles_to_xyz_block(mol.smiles)
            if xyz_block is None:
                mol.xtb_error = f"无法从 SMILES 生成 3D 构象: {mol.smiles}"
                return mol

            # 2. 在工作目录中运行 xtb
            if work_dir is None:
                tmpdir = tempfile.mkdtemp(prefix="xtb_")
                cleanup = True
            else:
                os.makedirs(work_dir, exist_ok=True)
                tmpdir = work_dir
                cleanup = False

            try:
                xyz_path = os.path.join(tmpdir, "input.xyz")
                write_xyz(xyz_block, xyz_path)

                env = {**os.environ, "OMP_NUM_THREADS": str(self.omp_num_threads)}
                cmd = [self.xtb_bin, xyz_path, "--gfn", str(self.gfn), "--opt"]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env=env,
                )

                if result.returncode != 0:
                    mol.xtb_error = f"xtb exited with code {result.returncode}: {result.stderr[-500:]}"
                    return mol

                stdout = result.stdout

                # 3. 解析输出
                parsed = parse_xtb_output(stdout)
                mol.xtb_gap = parsed.get("gap_eV")
                mol.xtb_energy = parsed.get("total_energy_Eh")

                # 解析 HOMO/LUMO 能量（用于相关性校准）
                orbitals = parse_xtb_orbital_energies(stdout)
                if orbitals is not None:
                    mol.xtb_homo, mol.xtb_lumo = orbitals

                if mol.xtb_gap is None:
                    mol.xtb_error = "无法从 xtb 输出中解析 HOMO-LUMO gap"
                    return mol

                # 4. 读取优化后的几何结构
                opt_xyz = os.path.join(tmpdir, "xtbopt.xyz")
                if os.path.exists(opt_xyz):
                    atomic_numbers, positions = read_xyz(opt_xyz)
                    mol.atomic_numbers = atomic_numbers
                    mol.positions = positions
                else:
                    mol.xtb_error = "xtb 优化后未找到 xtbopt.xyz 文件"
                    return mol

                mol.xtb_success = True

            finally:
                if cleanup:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)

        except subprocess.TimeoutExpired:
            mol.xtb_error = f"xtb 计算超时 ({self.timeout}s)"
        except FileNotFoundError:
            mol.xtb_error = f"xtb 二进制未找到: {self.xtb_bin}"
        except Exception as e:
            mol.xtb_error = f"xtb 运行时异常: {type(e).__name__}: {e}"

        return mol

    def run_batch(
        self, molecules: List[MoleculeData], work_dir_base: Optional[str] = None
    ) -> List[MoleculeData]:
        """批量运行 xtb（串行）.

        Args:
            molecules: MoleculeData 列表.
            work_dir_base: 工作目录基础路径, None 则每个分子独立临时目录.

        Returns:
            更新后的 MoleculeData 列表.
        """
        results = []
        for i, mol in enumerate(molecules):
            wdir = None
            if work_dir_base is not None:
                wdir = os.path.join(work_dir_base, f"mol_{i:04d}")
            results.append(self.run(mol, work_dir=wdir))
        return results
