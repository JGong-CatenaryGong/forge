# FORGE Reflection RAG

QM9 分子检索增强生成（RAG）组件 — 基于 FAISS 二值指纹索引 + parquet 元数据管理，支持实验实例隔离和评估反馈增量更新。

## 设计理念

RAG 的职责是**纯粹的分子信息检索**，不持有任务意图。它提供两样东西：

1. **结构化检索结果**（`query()` — 供程序化使用）
2. 原始数据交由上游 LLM 核心组件拼装 prompt

每个实验 run 从纯净 QM9 基座 fork，互不污染。

## 架构

```
                        QM9 基座 (124K 分子, 只读, 共享)
                     ┌── ├─ base_index.faissbin   (FAISS IndexBinaryFlat)
                     │   └─ base_metadata.parquet  (属性表)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Run #1        Run #2        Run #3
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Metadata │  │ Metadata │  │ Metadata │  ← 独立副本
  │  (copy)  │  │  (copy)  │  │  (copy)  │
  ├─────────┤  ├─────────┤  ├─────────┤
  │ 增量索引 │  │ 增量索引 │  │ 增量索引 │  ← IndexBinaryIDMap
  │ (空初始) │  │ (空初始) │  │ (空初始) │
  └─────────┘  └─────────┘  └─────────┘
```

- **基座**：一次性从 QM9 构建，序列化到 `data/rag_base/`
- **Fork**：每个 run 调用 `RAGEngine.fork()` 得到独立副本
- **更新**：`add_reflection()` 仅影响当前实例
- **持久化**：每实例可独立 save/load

## 安装

### 依赖

- Python ≥ 3.10
- `faiss-cpu ≥ 1.7` — 向量检索
- `pandas ≥ 1.3` — 元数据管理
- `pyarrow ≥ 10.0` — parquet 持久化
- `rdkit ≥ 2023.03` — 分子指纹
- [可选] `torch ≥ 1.12` — 基座构建（加载 QM9 预训练数据）

### 安装

```bash
cd molagent/reflection_rag
pip install -e .
```

## 快速开始

### 1. 构建基座（一次性）

```python
from reflection_rag import RAGEngine

RAGEngine.build_base()
```

首次运行约 4 分钟（130K 分子 → 指纹 → FAISS 索引）。数据缓存到 `~/study/molagent/data/rag_base/`，后续 fork 直接使用。

### 2. Fork 独立实例并查询

```python
rag = RAGEngine.fork()

# 按属性范围检索
results = rag.query(
    gap_range=(3.5, 4.5),
    homo_range=(-8.0, -5.0),
    top_k=20,
)

for r in results[:5]:
    print(f"  {r['smiles']:30s}  gap={r['gap_eV']:.2f}  homo={r['homo_eV']:.2f}")
```

### 3. 按指纹相似度检索

```python
# 以苯为参考，找结构相似的分子
results = rag.query(
    gap_range=(4.0, 8.0),
    smiles="c1ccccc1",   # 查询分子
    top_k=20,
)
```

### 4. 评估反馈更新

```python
# 接受 evaluator 的结果，追加到当前实例
rag.add_reflection(
    smiles_list=["c1ccccc1", "CC(=O)O"],
    properties=[
        {"homo_eV": -6.75, "lumo_eV": 0.14, "gap_eV": 6.89},
        {"homo_eV": -7.36, "lumo_eV": 0.46, "gap_eV": 7.82},
    ],
)
```

### 5. 实例隔离验证

```python
rag1 = RAGEngine.fork()
rag2 = RAGEngine.fork()

rag1.add_reflection(["CCO"], [{"gap_eV": 8.5}])

print(rag1)  # reflection=1
print(rag2)  # reflection=0  ← 不受影响
```

### 6. 持久化

```python
# 保存
rag.save("runs/exp_001/")

# 后续加载（会自动加载基座）
rag = RAGEngine.load("runs/exp_001/")
```

## API 参考

### `RAGEngine`

#### 类方法

| 方法 | 说明 |
|------|------|
| `build_base(max_heavy_atoms=None, force_rebuild=False)` | 构建 QM9 基座（一次性）。`max_heavy_atoms` 限制重原子数。 |
| `fork()` | 从基座 fork 独立实例。 |
| `load(directory)` | 从目录加载持久化实例。 |

#### 实例方法

| 方法 | 说明 |
|------|------|
| `query(homo_range, lumo_range, gap_range, dipole_range, n_atoms_max, smiles, top_k)` | 按属性范围 + 可选指纹相似度检索。返回结果列表。 |
| `add_reflection(smiles_list, properties)` | 追加评估反馈分子到当前实例（自动生成指纹并更新索引）。 |
| `save(directory)` | 保存当前实例到目录。 |

#### 属性

| 属性 | 说明 |
|------|------|
| `n_total` | 实例总分子数 |
| `n_reflection` | 通过 reflection 添加的分子数 |

#### `query()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `homo_range` | `(float, float)` | `None` | HOMO 范围 (eV) |
| `lumo_range` | `(float, float)` | `None` | LUMO 范围 (eV) |
| `gap_range` | `(float, float)` | `None` | HOMO-LUMO gap 范围 (eV) |
| `dipole_range` | `(float, float)` | `None` | 偶极矩范围 (Debye) |
| `n_atoms_max` | `int` | `None` | 最大原子数 |
| `smiles` | `str` | `None` | 查询分子 SMILES（用于指纹相似度排序） |
| `top_k` | `int` | `50` | 返回结果数 |

#### 返回结果字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `int` | 分子 ID |
| `smiles` | `str` | SMILES 字符串 |
| `homo_eV` | `float` | HOMO 能量 (eV) |
| `lumo_eV` | `float` | LUMO 能量 (eV) |
| `gap_eV` | `float` | HOMO-LUMO gap (eV) |
| `dipole_D` | `float` | 偶极矩 (Debye) |
| `n_atoms` | `int` | 原子数 |
| `source` | `str` | `"qm9"` 或 `"reflection"` |
| `distance` | `float` | 指纹距离（仅指纹查询时） |

### `MetadataStore`

分子属性元数据表，底层为 pandas DataFrame + parquet 持久化。

```python
from reflection_rag import MetadataStore

# 从 QM9 构建
meta = MetadataStore.from_qm9(qm9_data)

# 过滤
filtered = meta.filter(gap_range=(3.0, 7.0), homo_range=(-8.0, -5.0))

# 持久化
meta.save("metadata.parquet")
meta = MetadataStore.load("metadata.parquet")
```

### `BinaryIndex`

FAISS 二值指纹索引封装（Morgan fingerprint, radius=2, 2048-bit）。

### 指纹函数

```python
from reflection_rag import smiles_to_fingerprint, xyz_to_fingerprint

fp = smiles_to_fingerprint("c1ccccc1")   # → (2048,) uint8
fp = xyz_to_fingerprint(z, pos)          # → (2048,) uint8
```

## 数据存储

| 路径 | 内容 | 说明 |
|------|------|------|
| `~/study/molagent/data/rag_base/` | 基座索引 + 元数据 | 共享，只读 |
| `~/study/molagent/data/qm9/` | QM9 预训练数据 | 基座构建时读取 |

## 性能

| 操作 | 规模 | 耗时 |
|------|------|------|
| 基座构建 | 130K → 124K 分子 | ~4 min |
| Fork | 复制 124K 元数据 | <0.1 s |
| 属性查询 | 124K → 过滤 → top-k | <1 ms |
| 指纹查询 | 基座 + 增量合并检索 | <5 ms |
| add_reflection | 追加 N 个分子 | <0.1 s / 分子 |

## 与 evaluator 集成

```python
from evaluator import Evaluator
from reflection_rag import RAGEngine

# 每个实验 fork 独立 RAG
rag = RAGEngine.fork()

# 查询初始候选
candidates = rag.query(gap_range=(4.5, 5.5), top_k=100)

# 交给 evaluator 计算
evaluator = Evaluator(use_correlation=True)
smiles_list = [c["smiles"] for c in candidates]
df = evaluator.evaluate(smiles_list, target_gap=5.0, n_top=5)

# 反馈更新 RAG
rag.add_reflection(
    smiles_list=df["smiles"].tolist(),
    properties=df.to_dict("records"),
)

# 持久化本 run
rag.save("runs/exp_001/")
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_BASE_DIR` | `~/study/molagent/data/rag_base` | 基座目录 |
| `QM9_PATH` | `~/study/molagent/data/qm9/processed/qm9_v3.pt` | QM9 数据路径 |

## 许可

MIT
