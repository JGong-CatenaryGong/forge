"""FAISS 索引封装 — 二值指纹检索 + 增量更新.

架构:
  - 基座索引 (IndexBinaryFlat): 130K QM9 指纹, 只读共享
  - Fork 索引 (IndexBinaryIDMap): 每个实验实例的增量存储
  - 查询时合并检索两个索引, 取 top-k
"""

import logging
import os
from typing import List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

FP_BITS = 2048


class BinaryIndex:
    """FAISS 二值指纹索引封装."""

    def __init__(self, index: faiss.IndexBinary, is_base: bool = False):
        self.index = index
        self.is_base = is_base
        self._id_map: Optional[List[int]] = None

    @classmethod
    def build(cls, vectors: np.ndarray, ids: List[int]) -> "BinaryIndex":
        """从指纹向量数组构建基座索引.

        Args:
            vectors: (N, FP_BITS) uint8 数组 (每个元素 0/1).
            ids: 对应的元数据 ID 列表.

        Returns:
            BinaryIndex (is_base=True).
        """
        n = vectors.shape[0]
        # FAISS binary index expects packed bytes: 2048 bits -> 256 bytes/vector
        packed = np.packbits(vectors, axis=1)
        index = faiss.IndexBinaryFlat(int(FP_BITS))
        index.add(packed)
        idx = cls(index, is_base=True)
        idx._id_map = list(ids)
        logger.info(f"基座索引构建完成: {n} 个向量")
        return idx

    @classmethod
    def fork_empty(cls) -> "BinaryIndex":
        """创建空 fork 索引."""
        index = faiss.IndexBinaryIDMap(faiss.IndexBinaryFlat(int(FP_BITS)))
        idx = cls(index, is_base=False)
        idx._id_map = []
        return idx

    def add(self, vectors: np.ndarray, ids: List[int]) -> None:
        """向 fork 索引添加新向量."""
        if self.is_base:
            raise RuntimeError("基座索引不可修改, 只能在 fork 实例上添加")
        packed = np.packbits(vectors, axis=1)
        self.index.add_with_ids(packed, np.array(ids, dtype=np.int64))
        self._id_map.extend(ids)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索单个索引, 返回 (distances, ids)."""
        if self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])
        distances, ids = self.index.search(query, min(k, self.index.ntotal))
        return distances, ids

    def merge_search(
        self, base_index: "BinaryIndex", query: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """合并检索基座和自身, 返回 top-k.

        Args:
            base_index: 基座索引.
            query: (n_queries, FP_BITS/8) 查询向量 (已packed).
            k: 返回数.

        Returns:
            (merged_distances, merged_ids): 形状 (n_queries, k).
        """
        d_base, id_base = base_index.search(query, k)
        d_self, id_self = self.search(query, k)

        all_dists = []
        all_ids = []
        for i in range(query.shape[0]):
            dists = []
            mids = []
            if d_base.size > 0 and d_base.shape[1] > 0:
                valid = id_base[i] >= 0
                dists.extend(d_base[i][valid].tolist())
                mids.extend(id_base[i][valid].tolist())
            if d_self.size > 0 and d_self.shape[1] > 0:
                valid = id_self[i] >= 0
                dists.extend(d_self[i][valid].tolist())
                mids.extend(id_self[i][valid].tolist())
            if len(dists) == 0:
                all_dists.append(np.zeros(k, dtype=np.float32))
                all_ids.append(np.full(k, -1, dtype=np.int64))
            else:
                order = np.argsort(dists)[:k]
                all_dists.append(np.array(dists, dtype=np.float32)[order])
                all_ids.append(np.array(mids, dtype=np.int64)[order])
        return np.array(all_dists), np.array(all_ids)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index_binary(self.index, path)
        if self._id_map:
            np.save(path + ".idmap.npy", np.array(self._id_map, dtype=np.int64))
        logger.info(f"索引已保存: {path} ({self.index.ntotal} 个向量)")

    @classmethod
    def load(cls, path: str, is_base: bool = False) -> "BinaryIndex":
        index = faiss.read_index_binary(path)
        idx = cls(index, is_base=is_base)
        idmap_path = path + ".idmap.npy"
        if os.path.exists(idmap_path):
            idx._id_map = np.load(idmap_path).tolist()
        return idx

    def __len__(self) -> int:
        return self.index.ntotal
