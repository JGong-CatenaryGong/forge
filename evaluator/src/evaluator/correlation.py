"""XTB→DFT 相关性校准模型 (MLP + 分子指纹).

基于 QM9 数据集训练多层感知机 (MLP)，将 XTB 计算的 HOMO/LUMO/GAP
校准到 B3LYP/6-31G(2d,p) 水平的 DFT 值。

输入特征:
  - XTB 能级: xtb_homo, xtb_lumo, xtb_gap (3 维)
  - MACCS 指纹: 167 位
  - RDKit 分子描述符: 217 维

该模型同时学习 HOMO、LUMO、GAP 三个参数，最小化三者综合偏差。
"""

import logging
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---- 常量 ----
_QM9_PATH = os.path.expanduser("~/study/molagent/data/qm9/processed/qm9_v3.pt")
_CACHE_DIR = os.path.expanduser("~/study/molagent/data/xtb_cache")
_FEAT_CACHE_DIR = os.path.expanduser("~/study/molagent/data/feat_cache")
_MODEL_DIR = os.path.expanduser("~/study/molagent/data/correlation_v2")
_XTB_BIN = os.environ.get("XTB_BIN", "/home/jgong/calculations/xtb-dist/bin/xtb")
_XTB_THREADS = int(os.environ.get("XTB_OMP_NUM_THREADS", "4"))

_IDX_HOMO = 2
_IDX_LUMO = 3
_IDX_GAP = 4

_ATOMIC_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 18: "Ar", 35: "Br",
}


class CorrelationModel:
    """XTB→DFT 多目标校准模型 (MLP).

    用法:
        model = CorrelationModel.load_or_train(n_samples=500)
        calib = model.calibrate(xtb_homo=-6.5, xtb_lumo=-1.2, xtb_gap=5.3, smiles="CCO")
        # calib = {"homo": -6.8, "lumo": -1.0, "gap": 5.8}
    """

    def __init__(self):
        self.mlp: Optional[MLPRegressor] = None
        self.scaler_x: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self._trained = False
        self._feat_dim: Optional[int] = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ================================================================
    # 公共接口
    # ================================================================

    def calibrate(
        self,
        xtb_homo: float,
        xtb_lumo: float,
        xtb_gap: float,
        smiles: str,
    ) -> dict:
        """将 XTB 值校准为 DFT 预测值.

        Args:
            xtb_homo: XTB HOMO 能量 (eV).
            xtb_lumo: XTB LUMO 能量 (eV).
            xtb_gap: XTB HOMO-LUMO gap (eV).
            smiles: SMILES 字符串（用于分子指纹/描述符）.

        Returns:
            {"homo": 校准后HOMO, "lumo": 校准后LUMO, "gap": 校准后Gap} (全部 eV).
        """
        if not self._trained:
            raise RuntimeError("模型未训练或未加载")
        features = self._extract_features_from_smiles(smiles, xtb_homo, xtb_lumo, xtb_gap)
        if features is None:
            # 回退：只用XTB特征
            features = np.array([xtb_homo, xtb_lumo, xtb_gap], dtype=np.float64)
            if self._feat_dim and len(features) < self._feat_dim:
                features = np.pad(features, (0, self._feat_dim - len(features)), constant_values=0)
        X = features.reshape(1, -1)
        X_scaled = self.scaler_x.transform(X)
        y_scaled = self.mlp.predict(X_scaled)
        y = self.scaler_y.inverse_transform(y_scaled.reshape(1, -1))
        return {
            "homo": float(y[0, 0]),
            "lumo": float(y[0, 1]),
            "gap": float(y[0, 2]),
        }

    def train(
        self,
        n_samples: int = 500,
        random_seed: int = 42,
        xtb_timeout: int = 300,
        force_recompute: bool = False,
    ) -> dict:
        """加载数据、运行 XTB、提取特征、训练 MLP.

        Args:
            n_samples: QM9 采样数.
            random_seed: 随机种子.
            xtb_timeout: 单分子 XTB 超时(秒).
            force_recompute: 是否强制重新计算所有数据（忽略缓存）.

        Returns:
            训练统计.
        """
        # 1. 加载 QM9
        logger.info(f"加载 QM9: {_QM9_PATH}")
        qm9_data = torch.load(_QM9_PATH, map_location="cpu", weights_only=False)

        np.random.seed(random_seed)
        indices = np.random.choice(len(qm9_data), size=min(n_samples, len(qm9_data)), replace=False)
        logger.info(f"采样 {len(indices)} 个分子")

        # 2. 收集 XTB 结果 + DFT 参考值 + 分子特征
        os.makedirs(_CACHE_DIR, exist_ok=True)
        os.makedirs(_FEAT_CACHE_DIR, exist_ok=True)

        all_features = []
        dft_homo, dft_lumo, dft_gap = [], [], []
        n_success = 0
        n_failed = 0

        for i, idx in enumerate(indices):
            mol_data = qm9_data[idx]
            idx_int = int(idx)
            z = mol_data["z"].numpy()
            pos = mol_data["pos"].numpy()
            y = mol_data["y"].flatten().numpy()

            # DFT 参考值
            dft_h = float(y[_IDX_HOMO])
            dft_l = float(y[_IDX_LUMO])
            dft_g = float(y[_IDX_GAP])

            # XTB 计算（带缓存）
            xtb_result = self._run_xtb_with_cache(idx_int, z, pos, xtb_timeout, force_recompute)
            if xtb_result is None:
                n_failed += 1
                continue

            # 分子特征（带缓存）
            feat = self._get_features_with_cache(idx_int, z, pos, xtb_result, force_recompute)
            if feat is None:
                n_failed += 1
                continue

            all_features.append(feat)
            dft_homo.append(dft_h)
            dft_lumo.append(dft_l)
            dft_gap.append(dft_g)
            n_success += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(indices)} ({n_success} 成功, {n_failed} 失败)")

        logger.info(f"数据准备完成: {n_success} 成功, {n_failed} 失败")
        if n_success < 50:
            raise RuntimeError(f"成功样本太少 ({n_success}), 无法训练")

        # 3. 构建特征矩阵
        X = np.array(all_features, dtype=np.float64)
        Y = np.column_stack([dft_homo, dft_lumo, dft_gap]).astype(np.float64)
        self._feat_dim = X.shape[1]
        logger.info(f"特征维度: {X.shape[1]} (3 XTB + MACCS + RDKit descriptors)")
        logger.info(f"训练集: X={X.shape}, Y={Y.shape}")

        # 4. 标准化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_x.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y)

        # 5. 训练 MLP
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,         # L2 正则化
            batch_size="auto",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=random_seed,
            verbose=False,
        )
        logger.info("训练 MLP (hidden=256→128→64)...")
        self.mlp.fit(X_scaled, Y_scaled)

        # 6. 评估
        Y_pred_scaled = self.mlp.predict(X_scaled)
        Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
        stats = self._compute_stats(Y, Y_pred)


        logger.info(
            f"训练完成: R² HOMO={stats['r2_homo']:.4f}, "
            f"LUMO={stats['r2_lumo']:.4f}, GAP={stats['r2_gap']:.4f}, "
            f"MAE GAP={stats['mae_gap']:.4f} eV"
        )
        logger.info(f"MLP 迭代 {self.mlp.n_iter_} 次, best_loss={self.mlp.best_loss_}")

        self._trained = True
        return {"n_total": n_success, "n_xtb_success": n_success} | stats

    def save(self) -> None:
        """保存模型、标准化器."""
        if not self._trained:
            raise RuntimeError("模型未训练，无法保存")
        os.makedirs(_MODEL_DIR, exist_ok=True)
        joblib.dump(self.mlp, os.path.join(_MODEL_DIR, "mlp.joblib"))
        joblib.dump(self.scaler_x, os.path.join(_MODEL_DIR, "scaler_x.joblib"))
        joblib.dump(self.scaler_y, os.path.join(_MODEL_DIR, "scaler_y.joblib"))
        np.save(os.path.join(_MODEL_DIR, "feat_dim.npy"), np.array([self._feat_dim]))
        logger.info(f"模型已保存到 {_MODEL_DIR}")

    def load(self) -> None:
        """加载模型."""
        mp = os.path.join(_MODEL_DIR, "mlp.joblib")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"模型文件未找到: {mp}")
        self.mlp = joblib.load(mp)
        self.scaler_x = joblib.load(os.path.join(_MODEL_DIR, "scaler_x.joblib"))
        self.scaler_y = joblib.load(os.path.join(_MODEL_DIR, "scaler_y.joblib"))
        fd = os.path.join(_MODEL_DIR, "feat_dim.npy")
        if os.path.exists(fd):
            self._feat_dim = int(np.load(fd)[0])
        self._trained = True
        logger.info(f"模型已加载 ({_MODEL_DIR})")

    @classmethod
    def load_or_train(
        cls,
        n_samples: int = 500,
        force_retrain: bool = False,
        **kwargs,
    ) -> "CorrelationModel":
        """加载已有模型，或训练新模型."""
        model = cls()
        if not force_retrain and os.path.exists(os.path.join(_MODEL_DIR, "mlp.joblib")):
            try:
                model.load()
                logger.info("加载已有 MLP 模型成功")
                return model
            except Exception as e:
                logger.warning(f"加载失败: {e}, 重新训练")
        model.train(n_samples=n_samples, **kwargs)
        model.save()
        return model

    # ================================================================
    # 特征提取
    # ================================================================

    def _get_features_with_cache(
        self, idx: int, z: np.ndarray, pos: np.ndarray,
        xtb_result: dict, force: bool,
    ) -> Optional[np.ndarray]:
        """获取特征向量（带缓存）."""
        cache_file = os.path.join(_FEAT_CACHE_DIR, f"mol_{idx}.npz")
        if not force and os.path.exists(cache_file):
            try:
                return np.load(cache_file)["features"]
            except Exception:
                pass
        feat = self._extract_features_from_xyz(z, pos, xtb_result)
        if feat is not None:
            np.savez_compressed(cache_file, features=feat)
        return feat

    def _extract_features_from_xyz(
        self, z: np.ndarray, pos: np.ndarray, xtb_result: dict,
    ) -> Optional[np.ndarray]:
        """从 XYZ 坐标构建分子对象并提取指纹/描述符."""
        mol = self._xyz_to_mol(z, pos)
        if mol is None:
            return None
        return self._build_feature_vector(mol, xtb_result)

    def _extract_features_from_smiles(
        self, smiles: str,
        xtb_homo: float, xtb_lumo: float, xtb_gap: float,
    ) -> Optional[np.ndarray]:
        """从 SMILES 构建分子对象并提取指纹/描述符."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        xtb_result = {"homo": xtb_homo, "lumo": xtb_lumo, "gap": xtb_gap}
        return self._build_feature_vector(mol, xtb_result)

    @staticmethod
    def _build_feature_vector(mol, xtb_result: dict) -> np.ndarray:
        """组装特征向量: XTB(3) + MACCS(167) + RDKit Descriptors(217)."""
        from rdkit.Chem import MACCSkeys, Descriptors

        # XTB 能级
        xtb_feat = np.array([
            xtb_result["homo"], xtb_result["lumo"], xtb_result["gap"]
        ], dtype=np.float64)

        # MACCS 指纹 (167 位 → float)
        fp = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.array([float(fp.GetBit(i)) for i in range(fp.GetNumBits())], dtype=np.float64)

        # RDKit 描述符 (217 维, NaN→0)
        desc_dict = Descriptors.CalcMolDescriptors(mol)
        desc_vals = np.array([
            0.0 if v is None or (isinstance(v, float) and np.isnan(v))
            else float(v)
            for v in desc_dict.values()
        ], dtype=np.float64)

        return np.concatenate([xtb_feat, maccs, desc_vals])

    @staticmethod
    def _xyz_to_mol(z: np.ndarray, pos: np.ndarray):
        """从原子序数和坐标构建 RDKit 分子对象（带键感知）."""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdDetermineBonds

            lines = [f"{len(z)}", ""]
            for atom_z, coord in zip(z, pos):
                sym = _ATOMIC_SYMBOLS.get(int(atom_z), "X")
                lines.append(f"{sym:2s}  {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")

            xyz_block = "\n".join(lines)
            mol = Chem.MolFromXYZBlock(xyz_block)
            if mol is None:
                return None

            rdDetermineBonds.DetermineBonds(mol, charge=0)
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    # ================================================================
    # XTB 计算
    # ================================================================

    def _run_xtb_with_cache(
        self, idx: int, z: np.ndarray, pos: np.ndarray, timeout: int, force: bool,
    ) -> Optional[dict]:
        cache_file = os.path.join(_CACHE_DIR, f"mol_{idx}.npz")
        if not force and os.path.exists(cache_file):
            try:
                cached = np.load(cache_file)
                return {"homo": float(cached["homo"]), "lumo": float(cached["lumo"]), "gap": float(cached["gap"])}
            except Exception:
                pass
        result = self._run_xtb(z, pos, timeout)
        if result is not None:
            np.savez(cache_file, **result)
        return result

    def _run_xtb(self, z: np.ndarray, pos: np.ndarray, timeout: int) -> Optional[dict]:
        """对单个分子运行 GFN2-xTB 单点计算."""
        try:
            with tempfile.TemporaryDirectory(prefix="xtb_corr_") as tmpdir:
                xyz_path = os.path.join(tmpdir, "mol.xyz")
                with open(xyz_path, "w") as f:
                    f.write(f"{len(z)}\n\n")
                    for atomic_num, coord in zip(z, pos):
                        sym = _ATOMIC_SYMBOLS.get(int(atomic_num), "X")
                        f.write(f"{sym:2s}  {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

                env = {**os.environ, "OMP_NUM_THREADS": str(_XTB_THREADS)}
                result = subprocess.run(
                    [self._find_xtb(), xyz_path, "--gfn", "2"],
                    capture_output=True, text=True, timeout=timeout,
                    cwd=tmpdir, env=env,
                )
                if result.returncode != 0:
                    return None
                return self._parse_xtb_orbitals(result.stdout)
        except (subprocess.TimeoutExpired, Exception):
            return None

    @staticmethod
    def _find_xtb() -> str:
        if os.path.isfile(_XTB_BIN):
            return _XTB_BIN
        import shutil
        found = shutil.which("xtb")
        if found:
            return found
        raise FileNotFoundError("未找到 xtb 二进制")

    @staticmethod
    def _parse_xtb_orbitals(stdout: str) -> Optional[dict]:
        lines = stdout.split("\n")
        last_homo_idx = -1
        last_lumo_idx = -1
        for i, line in enumerate(lines):
            if "(HOMO)" in line:
                last_homo_idx = i
            if "(LUMO)" in line:
                last_lumo_idx = i
        if last_homo_idx < 0 or last_lumo_idx < 0:
            return None
        try:
            homo_parts = lines[last_homo_idx].split()
            homo_eV = float(homo_parts[homo_parts.index("(HOMO)") - 1])
            lumo_parts = lines[last_lumo_idx].split()
            lumo_eV = float(lumo_parts[lumo_parts.index("(LUMO)") - 1])
            gap_eV = None
            for line in reversed(lines):
                m = re.search(r"HOMO-LUMO\s+GAP\s+([\d.]+)\s+eV", line)
                if m:
                    gap_eV = float(m.group(1))
                    break
            if gap_eV is None:
                gap_eV = lumo_eV - homo_eV
            return {"homo": homo_eV, "lumo": lumo_eV, "gap": gap_eV}
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import r2_score, mean_absolute_error
        return {
            "r2_homo": round(float(r2_score(y_true[:, 0], y_pred[:, 0])), 4),
            "r2_lumo": round(float(r2_score(y_true[:, 1], y_pred[:, 1])), 4),
            "r2_gap": round(float(r2_score(y_true[:, 2], y_pred[:, 2])), 4),
            "mae_homo": round(float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])), 4),
            "mae_lumo": round(float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])), 4),
            "mae_gap": round(float(mean_absolute_error(y_true[:, 2], y_pred[:, 2])), 4),
        }
