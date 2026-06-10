# MolAgent 使用文档

## 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│  LLM Provider (openai 兼容 API)                                      │
│  └─ 调用 LLM 生成 SMILES                                             │
├──────────────────────────────────────────────────────────────────────┤
│  LLMCore / SimpleLLMCore — 主编排器                                  │
│  循环: RAG→Prompt→LLM→Evaluator→记录→反馈→RAG                       │
├──────────────────────────────────────────────────────────────────────┤
│  Evaluator / SimpleEvaluator — 分子评估                              │
│  Evaluator:      XTB(全部) → DFT(top N)                             │
│  SimpleEvaluator: XTB(全部), 无 DFT                                  │
├──────────────────────────────────────────────────────────────────────┤
│  ReflectionRAG — 分子检索 + 增量反馈索引                             │
│  基座: QM9 ~124K 分子 (FAISS 二值指纹)                               │
│  Fork 机制: 每个实验独立实例                                        │
├──────────────────────────────────────────────────────────────────────┤
│  ResultStore — 结果持久化                                            │
│  round_N/prompt.json, llm_response.txt, usage.json,                 │
│  eval_results.parquet, feedback_table.csv, summary.json              │
└──────────────────────────────────────────────────────────────────────┘
```

## 组件说明

### 1. `llm_core` / `llm_core_simple` — 主编排器

```python
from llm_core import LLMCore                    # DFT 版
from llm_core_simple import SimpleLLMCore       # XTB 纯版
```

两者接口一致：

```python
core = LLMCore(
    provider=provider,           # LLMProvider
    record_prompts=True,         # 是否保存 prompt 到磁盘
    reflection_mode="spr",       # 反馈模式: "spr" / "scalar"
    use_rag=True,                # 是否启用 RAG
)

result = core.run(
    target={"HOMO-LUMO gap": "3.0 eV"},  # 目标属性
    n_rounds=10,                          # 迭代轮数
    n_per_round=20,                       # 每轮生成分子数
    n_feedback=5,                         # 每轮反馈分子数
    evaluator=evaluator,                  # Evaluator / SimpleEvaluator
    rag=rag,                              # RAGEngine 实例
    run_name="my_experiment",             # 实验名称
    runs_dir="runs",                      # 结果根目录
    seed=42,                              # 随机种子
    max_tokens=16384,                     # LLM max_tokens
    temperature=0.8,                      # LLM 温度
    gap_range_margin=2.0,                 # RAG 检索 gap 范围 (±)
)
```

返回值:
```python
{
    "run_dir": "runs/my_experiment",       # 结果目录
    "best": {"smiles": "...", "dft_gap_eV": 3.0012, ...},  # 最佳分子
    "rounds": [...],                       # 每轮简要数据
    "failed": False,                       # 是否全失败
}
```

### 2. `evaluator` — DFT 评估器

```python
from evaluator import Evaluator

evaluator = Evaluator(
    n_parallel=4,               # 并行度
    use_correlation=True,       # XTB→DFT 相关性校准
    xtb_bin="/path/to/xtb",     # XTB 二进制路径
    dft_functional="B3LYP",     # DFT 泛函
    dft_basis="6-31G(2d,p)",    # DFT 基组
)
df = evaluator.evaluate(
    smiles_list=["CCO", "c1ccccc1"],
    target_gap=3.0,
    n_top=3,                    # 进入 DFT 的 top N 分子
)
# df 包含: smiles, xtb_gap_eV, dft_gap_eV, dft_homo_eV,
#          dft_lumo_eV, dft_dipole_D, dft_energy_Ha, ...
```

流程:
1. 对所有 n 个分子运行 XTB 几何优化
2. 可选: MLP 校准 XTB→DFT gap
3. 按 |gap - target| 排序，取 top N
4. 对 top N 运行 pySCF DFT 精确计算
5. 返回 DataFrame

### 3. `simple_evaluator` — 纯 XTB 评估器

```python
from simple_evaluator import SimpleEvaluator

evaluator = SimpleEvaluator(
    n_parallel=4,               # 并行度
    xtb_bin="/path/to/xtb",     # XTB 二进制路径
)
df = evaluator.evaluate(
    smiles_list=["CCO", "c1ccccc1"],
    target_gap=3.0,
    n_top=3,
)
# df 包含: smiles, xtb_gap_eV, xtb_homo_eV, xtb_lumo_eV,
#          xtb_dipole_D, xtb_energy_Ha, ...
```

流程:
1. 对所有 n 个分子运行 XTB 几何优化 (含 --dipole)
2. 按 |xtb_gap - target| 排序，取 top N
3. 返回 DataFrame

无 DFT、无相关性校准。

### 4. `reflection_rag` — RAG 检索引擎

```python
from reflection_rag import RAGEngine

# 一次性构建基座 (已有可跳过)
RAGEngine.build_base()          # 加载 QM9 ~124K 分子

# 每个实验 fork 独立实例
rag = RAGEngine.fork()          # 拷贝基座，不污染它

# RAG 检索
results = rag.query(
    feedbacks=[...],            # 历史反馈
    n_samples=20,               # 需要的样本数
    gap_range_margin=2.0,       # gap 检索范围
)
# results: [{smiles, xtb_gap, ...}, ...]

# 更新反馈 (每轮后)
rag.add_reflection(
    smiles_list=["..."],        # 本轮成功分子
    properties_list=[{...}],    # 对应属性字典列表
)
```

### 5. `llm_core.providers` — LLM 调用

```python
from llm_core.providers import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-v4-flash",
    timeout=300,                # HTTP 超时 (秒)
)
```

内置重试: 429/503/timeout 自动重试 3 次 (4s, 8s 退避)。

### 6. `experiment.models_config` — 模型配置加载

```python
from experiment.models_config import load_models, run_dir_name

models = load_models()          # 从 api_llm.txt 加载
cfg = models[0]                 # ModelConfig(model, api_key, base_url, provider, timeout)

name = run_dir_name(cfg)        # → "DeepSeek_v4-flash"
```

## 如何写一个 run.py

### 标准模式 (DFT)

```python
import logging, sys, os
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from llm_core import LLMCore
from llm_core.providers import OpenAICompatibleProvider
from evaluator import Evaluator
from reflection_rag import RAGEngine

CONFIG = {
    "target": {"HOMO-LUMO gap": "3.0 eV"},
    "n_rounds": 10,
    "n_per_round": 20,
    "n_feedback": 5,
    "max_tokens": 16384,
    "temperature": 0.8,
    "gap_range_margin": 2.0,
    "reflection_mode": "spr",
    "use_rag": True,
    "seeds": [42, 123, 456],
}

# 加载模型
from experiment.models_config import load_models
cfg = [m for m in load_models() if m.model == "deepseek-v4-flash"][0]

provider = OpenAICompatibleProvider(
    api_key=cfg.api_key, base_url=cfg.base_url,
    model=cfg.model, timeout=cfg.timeout,
)
evaluator = Evaluator(n_parallel=4, use_correlation=True)
RAGEngine.build_base()

for seed in CONFIG["seeds"]:
    rag = RAGEngine.fork()
    core = LLMCore(provider=provider, reflection_mode=CONFIG["reflection_mode"],
                   use_rag=CONFIG["use_rag"])
    result = core.run(
        target=CONFIG["target"],
        n_rounds=CONFIG["n_rounds"],
        n_per_round=CONFIG["n_per_round"],
        n_feedback=CONFIG["n_feedback"],
        evaluator=evaluator,
        rag=rag,
        run_name=f"my_exp/seed_{seed}",
        seed=seed,
        max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        gap_range_margin=CONFIG["gap_range_margin"],
    )
    print(f"seed {seed}: best={result['best'].get('dft_gap_eV')}")
```

### Simple 模式 (纯 XTB)

```python
import logging, sys, os
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

# 只需替换两个 import
from llm_core_simple import SimpleLLMCore    # ← 换
from simple_evaluator import SimpleEvaluator # ← 换
from llm_core.providers import OpenAICompatibleProvider
from reflection_rag import RAGEngine
from experiment.models_config import load_models

cfg = [m for m in load_models() if m.model == "deepseek-v4-flash"][0]
provider = OpenAICompatibleProvider(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model)

evaluator = SimpleEvaluator(n_parallel=4)    # ← XTB only, 无 DFT
RAGEngine.build_base()

for seed in [42, 123, 456]:
    rag = RAGEngine.fork()
    core = SimpleLLMCore(provider=provider)  # ← SimpleLLMCore
    result = core.run(
        target={"HOMO-LUMO gap": "3.0 eV"},
        n_rounds=10, n_per_round=20, n_feedback=5,
        evaluator=evaluator,                 # ← SimpleEvaluator
        rag=rag,
        run_name=f"simple/seed_{seed}",
        seed=seed, max_tokens=16384, temperature=0.8,
    )
```

## 标准模式 vs Simple 模式对比

| 维度 | 标准 (llm_core + Evaluator) | Simple (llm_core_simple + SimpleEvaluator) |
|---|---|---|
| **评估流程** | XTB(全部) → DFT(top N) | XTB(全部), 无 DFT |
| **反馈字段** | DFT gap/HOMO/LUMO/偶极/能量 | **XTB** gap/HOMO/LUMO/偶极/能量 |
| **字段前缀** | `dft_gap_eV`, `dft_homo_eV`, ... | `xtb_gap_eV`, `xtb_homo_eV`, ... (SimpleLLMCore 自动映射为 dft_*) |
| **速度** | 慢 (DFT 每分子 1-5 min) | 快 (XTB 每分子 <1s) |
| **成本** | DFT 计算资源高 | 仅 XTB |
| **相关性校准** | 可选 (MLP) | 无 |
| **反馈延迟** | 高 (等 DFT 完成) | 低 (XTB 秒级) |
| **精确度** | DFT gap (接近真实) | XTB gap (近似, 趋势一致) |
| **适用场景** | 需要精确 DFT 值的最终验证 | 快速迭代、gap 扫描、方法对比 |

### SimpleLLMCore 的字段映射

SimpleLLMCore 内部自动将 `xtb_*` 映射为 `dft_*`，使下游 prompts / result_store / RAG 兼容：

```
xtb_gap_eV      → dft_gap_eV
xtb_homo_eV     → dft_homo_eV
xtb_lumo_eV     → dft_lumo_eV
xtb_dipole_D    → dft_dipole_D
xtb_energy_Ha   → dft_energy_Ha
xtb_success     → dft_success
```

所以用 SimpleLLMCore 跑出的结果目录结构与标准版完全一致，可以直接用 `summarize.py` 分析。

## 结果目录结构

```
runs/my_experiment/
├── manifest.json              # 总览 (配置, 轮次概要)
├── summary.json               # 最佳分子 + 汇总统计
├── feedback_table.csv         # 累积反馈表
├── round_01/
│   ├── prompt.json            # 发送给 LLM 的完整 prompt
│   ├── llm_response.txt       # LLM 原始回复
│   ├── usage.json             # token 用量
│   ├── generated_smiles.txt   # 解析出的 SMILES 列表
│   ├── eval_results.parquet   # 评估结果 (DataFrame)
│   └── rag_context.json       # RAG 检索到的参考分子
├── round_02/ ...
└── round_10/ ...
```

## 模型配置 (api_llm.txt)

格式: `model_name, api_key, base_url`

```
deepseek-v4-flash,sk-xxx,https://api.deepseek.com/v1
kimi-k2.6,sk-xxx,https://api.moonshot.cn/v1
glm-5.1,sk-xxx,https://api.glm.cn/v4
...
```

注意: kimi-k2.6 需要 temperature=1.0 (已在 `MODEL_TEMPERATURE` 中预设)。

## 快速参考

```bash
# 标准 DFT 实验
python experiment/run.py

# 多模型对比
python experiment/run_compare.py

# 单模型 gap 扫描 (XTB)
python experiment/run_gap_sweep.py --model deepseek-v4-flash

# per-molecule 模式 (单分子高密度反思)
python experiment/run_permolecule.py

# 结果分析
python experiment/summarize.py
```
