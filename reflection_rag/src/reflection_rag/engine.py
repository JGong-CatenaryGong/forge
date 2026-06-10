"""RAGEngine — QM9 分子检索 + 实例隔离 + 评估反馈更新.

核心设计:
  - 基座 (base): 从 QM9 构建，只读，一次构建多次复用
  - Fork: 每个实验从基座 fork，拥有独立的增量索引和元数据
  - 更新: 接受 evaluator 的 DFT 结果，追加到当前实例
  - 持久化: 每个 run 可独立保存/加载

用法:
    # 一次性构建基座
    RAGEngine.build_base()

    # 每个实验 fork 独立实例
    rag = RAGEngine.fork()
    results = rag.query(gap_range=(3.0, 7.0), top_k=50)

    # 更新反馈
    rag.add_reflection(smiles_list, properties_list)

    # 持久化
    rag.save("runs/exp_001/")
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fingerprints import (
    xyz_to_fingerprint,
    xyz_to_smiles,
    smiles_to_fingerprint,
    FP_BITS,
)
from .metadata import MetadataStore
from .indexer import BinaryIndex

logger = logging.getLogger(__name__)

# 基座数据路径
_BASE_DIR = os.path.expanduser("~/study/molagent/data/rag_base")
_BASE_INDEX_PATH = os.path.join(_BASE_DIR, "base_index.faissbin")
_BASE_META_PATH = os.path.join(_BASE_DIR, "base_metadata.parquet")
_QM9_PATH = os.path.expanduser("~/study/molagent/data/qm9/processed/qm9_v3.pt")


class RAGEngine:
    """RAG 引擎 — QM9 检索 + 增量更新."""

    _base_index: Optional[BinaryIndex] = None
    _base_metadata: Optional[MetadataStore] = None

    def __init__(self, metadata: MetadataStore, index: BinaryIndex, base_index: BinaryIndex):
        self.metadata = metadata
        self.index = index
        self.base_index = base_index

    # ================================================================
    # 基座构建 (一次性)
    # ================================================================

    @classmethod
    def build_base(
        cls,
        max_heavy_atoms: Optional[int] = None,
        force_rebuild: bool = False,
    ) -> None:
        """构建 QM9 基座（索引 + 元数据），保存到磁盘.

        Args:
            max_heavy_atoms: 最大重原子数 (None=全部). QM9 不超过 9 重原子.
            force_rebuild: 强制重建.
        """
        if not force_rebuild and os.path.exists(_BASE_INDEX_PATH) and os.path.exists(_BASE_META_PATH):
            logger.info("基座已存在，跳过构建")
            return

        import torch
        logger.info(f"加载 QM9: {_QM9_PATH}")
        qm9_data = torch.load(_QM9_PATH, map_location="cpu", weights_only=False)

        # 1. 构建 metadata — 不限制总原子数
        meta = MetadataStore.from_qm9(qm9_data, max_atoms=None)

        # 2. 生成指纹并构建索引
        logger.info(f"生成指纹 (共 {len(meta)} 个分子)...")
        fp_vectors = []
        valid_ids = []
        for i, idx in enumerate(meta.df["id"].tolist()):
            if (i + 1) % 20000 == 0:
                logger.info(f"  指纹进度: {i+1}/{len(meta)}")
            mol_data = qm9_data[idx]
            z = mol_data["z"].numpy()
            pos = mol_data["pos"].numpy()
            fp = xyz_to_fingerprint(z, pos)
            if fp is None:
                continue
            smiles = xyz_to_smiles(z, pos)
            if smiles:
                meta.df.at[i, "smiles"] = smiles
            fp_vectors.append(fp)
            valid_ids.append(idx)

        # 只保留成功解析的分子
        meta.df = meta.df[meta.df["id"].isin(valid_ids)].reset_index(drop=True)
        fp_array = np.array(fp_vectors, dtype=np.uint8)

        index = BinaryIndex.build(fp_array, meta.get_ids())
        os.makedirs(_BASE_DIR, exist_ok=True)
        index.save(_BASE_INDEX_PATH)
        meta.save(_BASE_META_PATH)
        logger.info(f"基座构建完成: {len(meta)} 个分子")

    # ================================================================
    # Fork / 实例管理
    # ================================================================

    @classmethod
    def fork(cls) -> "RAGEngine":
        """从基座 fork 一个独立实例."""
        cls._ensure_base_loaded()
        meta = cls._base_metadata.copy()
        fork_index = BinaryIndex.fork_empty()
        logger.info(f"Fork 完成: metadata={len(meta)} records")
        return cls(metadata=meta, index=fork_index, base_index=cls._base_index)

    @classmethod
    def _ensure_base_loaded(cls) -> None:
        if cls._base_index is None:
            cls._base_index = BinaryIndex.load(_BASE_INDEX_PATH, is_base=True)
        if cls._base_metadata is None:
            cls._base_metadata = MetadataStore.load(_BASE_META_PATH)

    # ================================================================
    # 查询
    # ================================================================

    def query(
        self,
        homo_range: Optional[Tuple[float, float]] = None,
        lumo_range: Optional[Tuple[float, float]] = None,
        gap_range: Optional[Tuple[float, float]] = None,
        dipole_range: Optional[Tuple[float, float]] = None,
        n_atoms_max: Optional[int] = None,
        smiles: Optional[str] = None,
        top_k: int = 50,
    ) -> List[Dict]:
        """按属性范围 + 指纹相似度检索分子."""
        # 1. 属性过滤
        filtered = self.metadata.filter(
            homo_range=homo_range, lumo_range=lumo_range,
            gap_range=gap_range, dipole_range=dipole_range,
            n_atoms_max=n_atoms_max,
        )

        if len(filtered) == 0:
            return []

        # 2. 如果提供了 SMILES，用指纹相似度排序
        if smiles is not None:
            fp = smiles_to_fingerprint(smiles)
            if fp is not None:
                fp_packed = np.packbits(fp.reshape(1, -1), axis=1)
                filtered_ids = set(filtered.get_ids())
                dists, ids = self.index.merge_search(
                    self.base_index, fp_packed, k=min(top_k * 3, len(self.metadata))
                )
                results = []
                for d, mid in zip(dists[0], ids[0]):
                    if mid in filtered_ids and mid >= 0:
                        rec = self.metadata.get_by_id(mid)
                        if rec:
                            rec["distance"] = float(d)
                            results.append(rec)
                    if len(results) >= top_k:
                        break
                return results

        # 3. 无指纹查询: 返回过滤结果前 top_k
        return filtered.to_records()[:top_k]

    # ================================================================
    # 更新 (Reflection)
    # ================================================================

    def add_reflection(
        self,
        smiles_list: List[str],
        properties: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """添加评估反馈分子到当前实例."""
        new_ids = self.metadata.add_reflection(smiles_list, properties)

        fp_list = []
        valid_ids = []
        for new_id, smi in zip(new_ids, smiles_list):
            fp = smiles_to_fingerprint(smi)
            if fp is not None:
                fp_list.append(fp)
                valid_ids.append(new_id)

        if fp_list:
            fp_array = np.array(fp_list, dtype=np.uint8)
            self.index.add(fp_array, valid_ids)
            logger.info(f"索引更新: {len(valid_ids)} 个新向量")

    # ================================================================
    # 持久化
    # ================================================================

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.metadata.save(os.path.join(directory, "metadata.parquet"))
        self.index.save(os.path.join(directory, "incremental_index.faissbin"))
        logger.info(f"实例已保存: {directory}")

    @classmethod
    def load(cls, directory: str) -> "RAGEngine":
        cls._ensure_base_loaded()
        meta = MetadataStore.load(os.path.join(directory, "metadata.parquet"))
        idx_path = os.path.join(directory, "incremental_index.faissbin")
        idx = BinaryIndex.load(idx_path, is_base=False) if os.path.exists(idx_path) else BinaryIndex.fork_empty()
        return cls(metadata=meta, index=idx, base_index=cls._base_index)

    # ================================================================
    # 状态
    # ================================================================

    @property
    def n_total(self) -> int:
        return len(self.metadata)

    @property
    def n_reflection(self) -> int:
        return int((self.metadata.df["source"] == "reflection").sum())

    def __repr__(self) -> str:
        return f"RAGEngine(total={self.n_total}, qm9={self.n_total - self.n_reflection}, reflection={self.n_reflection})"
