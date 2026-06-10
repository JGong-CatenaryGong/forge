# MolAgent Evaluator

分子属性评估系统：通过 **XTB 半经验预筛选** + **[可选 ML 校准]** + **pySCF DFT 精确计算** 三级流水线，从大量候选 SMILES 中筛选出最接近目标 HOMO-LUMO gap 的分子。

## 架构

```
候选SMILES₁…ₙ
  │
  ▼
┌──────────────────────────────────────────────┐
│ 阶段1: XTB (GFN2-xTB) 几何优化                │
│   · RDKit 3D构象生成 → xtb --gfn 2 --opt     │
│   · 输出: xtb_gap, xtb_homo, xtb_lumo,       │
│           优化后坐标                           │
├──────────────────────────────────────────────┤
│ 阶段1.5: [可选] ML 相关性校准                  │
│   · 输入: xtb能级 + MACCS指纹 + RDKit描述符   │
│   · 模型: MLP(256→128→64), QM9训练            │
│   · 输出: calib_gap (MAE ≈ 0.09 eV)           │
├──────────────────────────────────────────────┤
│ 阶段2: 按 |gap - target| 排序，选取 Top-N     │
├──────────────────────────────────────────────┤
│ 阶段3: pySCF DFT 精确计算                     │
│   · 默认: B3LYP / 6-31G(2d,p)                │
│   · 输出: dft_gap, dft_homo, dft_lumo,       │
│           dft_dipole, dft_mulliken_charge,    │
│           dft_energy                          │
└──────────────────────────────────────────────┘
  │
  ▼
结构化 DataFrame 输出
```

## 安装

### 依赖

- **Python** ≥ 3.10
- **xtb** 二进制 (GFN2-xTB)，需在 PATH 中或通过 `XTB_BIN` 环境变量指定
- Python 包: `rdkit`, `pyscf`, `numpy`, `pandas`, `scikit-learn`, `joblib`
- [可选] 相关性校准: `torch` (用于加载 QM9 预训练数据)

### 从源码安装

```bash
cd molagent
pip install -e .
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `XTB_BIN` | `/home/jgong/calculations/xtb-dist/bin/xtb` | xtb 二进制路径 |
| `XTB_OMP_NUM_THREADS` | `4` | 每个 xtb 进程的 OpenMP 线程数 |

## 快速开始

### 基础用法（无校准）

```python
from evaluator import Evaluator

evaluator = Evaluator(n_parallel=4)

df = evaluator.evaluate(
    smiles_list=["CCO", "CCN", "CC(=O)O", "c1ccccc1", "C1CCCCC1"],
    target_gap=5.0,   # 目标 HOMO-LUMO gap (eV)
    n_top=3,          # 最终输出前 3 个分子
)

print(df)
#    smiles     xtb_gap_eV  calib_gap_eV  dft_gap_eV  dft_homo_eV  dft_lumo_eV  dft_dipole_D  ...
# 0  c1ccccc1      4.9343         None       6.8908      -6.7535       0.1373        0.0000  ...
# 1  CC(=O)N       5.2057         None       7.6630      -7.3145       0.3485        3.7013  ...
# 2  CC(=O)O       5.2117         None       7.8233      -7.3586       0.4646        1.6819  ...
```

### 启用 ML 相关性校准

```python
evaluator = Evaluator(
    use_correlation=True,          # 启用 XTB→DFT 校准
    correlation_n_samples=500,     # QM9 训练样本数
    n_parallel=4,
)

df = evaluator.evaluate(smiles_list, target_gap=5.0, n_top=3)
```

首次运行时会自动下载 QM9 数据（~54MB）并训练 MLP 模型（~1分钟, 500样本）。后续运行直接加载已训练的模型。

### 自定义 DFT 计算参数

```python
evaluator = Evaluator(
    dft_functional="PBE0",         # 泛函
    dft_basis="def2-SVP",          # 基组
    dft_charge=0,                  # 电荷
    dft_spin=0,                    # 自旋 (0=闭壳层单重态)
    dft_max_scf_cycles=200,        # SCF 最大迭代
)
```

### 自定义 XTB 参数

```python
evaluator = Evaluator(
    xtb_gfn=2,                     # GFN 版本 (0/1/2)
    xtb_omp_threads=8,             # OpenMP 线程数
    xtb_timeout=600,               # 单分子超时 (秒)
    xtb_bin="/path/to/xtb",        # xtb 路径
)
```

## API 参考

### `Evaluator`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `xtb_bin` | `str \| None` | `None` | xtb 二进制路径 |
| `xtb_gfn` | `int` | `2` | GFN 方法 (0/1/2) |
| `xtb_omp_threads` | `int` | `4` | 每进程 OpenMP 线程 |
| `xtb_timeout` | `int` | `300` | 单分子超时 (秒) |
| `dft_functional` | `str` | `"B3LYP"` | DFT 泛函 |
| `dft_basis` | `str` | `"6-31G(2d,p)"` | 基组 |
| `dft_charge` | `int` | `0` | 分子电荷 |
| `dft_spin` | `int` | `0` | 自旋 (0=闭壳层) |
| `dft_max_scf_cycles` | `int` | `200` | SCF 最大迭代 |
| `n_parallel` | `int` | `4` | 并行分子数 |
| `use_correlation` | `bool` | `False` | 启用 ML 校准 |
| `correlation_n_samples` | `int` | `500` | 校准模型训练样本数 |
| `correlation_force_retrain` | `bool` | `False` | 强制重训校准模型 |

**`evaluate(smiles_list, target_gap, n_top) → pd.DataFrame`**

| 参数 | 类型 | 说明 |
|------|------|------|
| `smiles_list` | `list[str]` | 候选 SMILES |
| `target_gap` | `float` | 目标 HOMO-LUMO gap (eV) |
| `n_top` | `int` | 最终输出的分子数 |
| `work_dir` | `str \| None` | 工作目录 (None=临时目录) |

**输出 DataFrame 字段：**

| 字段 | 单位 | 说明 |
|------|------|------|
| `smiles` | — | SMILES 字符串 |
| `xtb_gap_eV` | eV | XTB HOMO-LUMO gap |
| `xtb_homo_eV` | eV | XTB HOMO 能量 |
| `xtb_lumo_eV` | eV | XTB LUMO 能量 |
| `xtb_deviation_eV` | eV | 与 target 的偏差 (用于排序) |
| `calib_gap_eV` | eV | 校准后 gap (仅 correlation=ON) |
| `calib_homo_eV` | eV | 校准后 HOMO (仅 correlation=ON) |
| `calib_lumo_eV` | eV | 校准后 LUMO (仅 correlation=ON) |
| `dft_gap_eV` | eV | DFT HOMO-LUMO gap |
| `dft_homo_eV` | eV | DFT HOMO 能量 |
| `dft_lumo_eV` | eV | DFT LUMO 能量 |
| `dft_dipole_D` | Debye | 偶极矩 |
| `dft_energy_Ha` | Hartree | 总能量 |
| `dft_mulliken_charge` | e | Mulliken 电荷 (每原子列表) |
| `dft_deviation_eV` | eV | DFT gap 与 target 偏差 |
| `target_gap_eV` | eV | 目标 gap |

### `CorrelationModel`

独立使用的 ML 校准模型。

```python
from evaluator import CorrelationModel

model = CorrelationModel.load_or_train(n_samples=500)

# 校准
calib = model.calibrate(
    xtb_homo=-10.95, xtb_lumo=-6.02, xtb_gap=4.93,
    smiles="c1ccccc1"
)
# → {"homo": -6.78, "lumo": 0.18, "gap": 6.96}
```

### `MoleculeData`

数据类，承载单个分子全流程的计算结果。包含 `to_dict()` 方法。

## 模型性能

### ML 校准模型 (MLP)

在 467 个 QM9 分子上的训练集性能：

| 参数 | R² | MAE (eV) | 斜率 |
|------|:--:|:--------:|:----:|
| HOMO | 0.97 | 0.05 | 0.98 |
| LUMO | 0.98 | 0.10 | 0.98 |
| GAP | 0.98 | 0.09 | 0.98 |

**模型特征 (387维):** XTB能级 (3) + MACCS指纹 (167) + RDKit描述符 (217)

**网络结构:** MLP(256→128→64 ReLU, L2=0.001, early stopping)

## 注意事项

1. **xtb 二进制**: 需要系统中安装 GFN2-xTB。可通过 `XTB_BIN` 环境变量或构造函数参数指定路径。
2. **计算耗时**: XTB 几何优化每个分子约 1-5 秒 (取决于分子大小和并行度)，pySCF DFT 每个分子约 5-30 秒。
3. **校准模型训练**: 首次启用时需要 ~1 分钟下载 QM9 并训练。模型缓存于 `~/molagent/data/`。
4. **无效 SMILES**: 自动跳过并标记错误，不影响其他分子的计算。
5. **并行**: 使用 `ThreadPoolExecutor`，实际并行度受 xtb 内部 OpenMP 和 CPU 核数制约。

## 许可

MIT
