"""FORGE RAG Engine — QM9 分子检索与增量更新.

基于 FAISS 二值指纹索引 + pandas/parquet 元数据管理.
支持实例隔离 (fork 模式)，每个实验独立运行互不污染.

用法:
    from reflection_rag import RAGEngine

    # 一次性构建基座
    RAGEngine.build_base()

    # 每个实验 fork 独立实例
    rag = RAGEngine.fork()
    results = rag.query(gap_range=(3.0, 7.0), top_k=50)

    # 更新评估反馈
    rag.add_reflection(smiles_list, properties_list)
"""

from .engine import RAGEngine
from .metadata import MetadataStore
from .indexer import BinaryIndex
from .fingerprints import smiles_to_fingerprint, xyz_to_fingerprint

__all__ = [
    "RAGEngine",
    "MetadataStore",
    "BinaryIndex",
    "smiles_to_fingerprint",
    "xyz_to_fingerprint",
]
