"""Metadata 管理 — QM9 属性表 + 增量更新."""

import logging
import os
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# QM9 属性索引
_IDX_HOMO = 2
_IDX_LUMO = 3
_IDX_GAP = 4
_IDX_DIPOLE = 0

DEFAULT_COLUMNS = [
    "id", "smiles", "homo_eV", "lumo_eV", "gap_eV", "dipole_D",
    "n_atoms", "source",  # source: "qm9" | "reflection"
]


class MetadataStore:
    """分子属性元数据表，基座于 pandas DataFrame + parquet 持久化.

    职责:
    - 从 QM9 构建初始表
    - 追加评估反馈的新分子
    - 按属性范围过滤查询
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else pd.DataFrame(columns=DEFAULT_COLUMNS)

    @classmethod
    def from_qm9(cls, qm9_data, max_atoms: Optional[int] = None) -> "MetadataStore":
        """从 QM9 PyG 数据集构建 MetadataStore.

        Args:
            qm9_data: QM9 PyG 数据集 (list of dicts).
            max_atoms: 最大原子数限制 (None = 全部).
        """
        records = []
        for idx, mol_data in enumerate(qm9_data):
            z = mol_data["z"].numpy()
            if max_atoms is not None and len(z) > max_atoms:
                continue
            y = mol_data["y"].flatten().numpy()
            records.append({
                "id": idx,
                "smiles": "",  # 延迟填充 (指纹生成时才获取SMILES)
                "homo_eV": float(y[_IDX_HOMO]),
                "lumo_eV": float(y[_IDX_LUMO]),
                "gap_eV": float(y[_IDX_GAP]),
                "dipole_D": float(y[_IDX_DIPOLE]),
                "n_atoms": int(len(z)),
                "source": "qm9",
            })
        df = pd.DataFrame(records)
        logger.info(f"MetadataStore 构建完成: {len(df)} 条 QM9 记录")
        return cls(df)

    def copy(self) -> "MetadataStore":
        """深拷贝（用于 fork）."""
        return MetadataStore(self.df.copy())

    def add_reflection(
        self,
        smiles_list: List[str],
        properties: Optional[List[Dict[str, float]]] = None,
        feedback: Optional[str] = None,
    ) -> List[int]:
        """追加评估反馈的新分子.

        Args:
            smiles_list: 新分子 SMILES 列表.
            properties: 对应的属性字典列表, 每个包含 homo_eV/lumo_eV/gap_eV/dipole_D 等.
            feedback: 关联的反思文本.

        Returns:
            新分配的 ID 列表.
        """
        next_id = int(self.df["id"].max()) + 1 if len(self.df) > 0 else 0
        new_ids = []
        for i, smi in enumerate(smiles_list):
            rec = {"id": next_id + i, "smiles": smi, "source": "reflection"}
            if properties and i < len(properties) and properties[i]:
                rec.update({k: v for k, v in properties[i].items() if k in DEFAULT_COLUMNS})
            new_ids.append(next_id + i)
        new_df = pd.DataFrame(new_ids)  # placeholder
        # Build properly
        new_rows = []
        for rid in new_ids:
            idx = rid - next_id
            row = {"id": rid, "smiles": smiles_list[idx], "source": "reflection"}
            if properties and idx < len(properties) and properties[idx]:
                for k, v in properties[idx].items():
                    if k in DEFAULT_COLUMNS:
                        row[k] = v
            new_rows.append(row)
        self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
        logger.info(f"MetadataStore 追加 {len(smiles_list)} 条 reflection 记录")
        return new_ids

    def filter(
        self,
        homo_range: Optional[tuple] = None,
        lumo_range: Optional[tuple] = None,
        gap_range: Optional[tuple] = None,
        dipole_range: Optional[tuple] = None,
        n_atoms_max: Optional[int] = None,
    ) -> "MetadataStore":
        """按属性范围过滤，返回新的 MetadataStore（不修改自身）."""
        mask = pd.Series(True, index=self.df.index)
        if homo_range:
            mask &= self.df["homo_eV"].between(*homo_range)
        if lumo_range:
            mask &= self.df["lumo_eV"].between(*lumo_range)
        if gap_range:
            mask &= self.df["gap_eV"].between(*gap_range)
        if dipole_range:
            mask &= self.df["dipole_D"].between(*dipole_range)
        if n_atoms_max is not None:
            mask &= self.df["n_atoms"] <= n_atoms_max
        return MetadataStore(self.df[mask].copy())

    def get_ids(self) -> List[int]:
        return self.df["id"].tolist()

    def get_by_id(self, mol_id: int) -> Optional[Dict]:
        rows = self.df[self.df["id"] == mol_id]
        if len(rows) == 0:
            return None
        return rows.iloc[0].to_dict()

    def to_records(self) -> List[Dict]:
        return self.df.to_dict("records")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.df.to_parquet(path, index=False)
        logger.info(f"MetadataStore 已保存: {path} ({len(self.df)} 条)")

    @classmethod
    def load(cls, path: str) -> "MetadataStore":
        df = pd.read_parquet(path)
        return cls(df)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"MetadataStore({len(self.df)} records)"
