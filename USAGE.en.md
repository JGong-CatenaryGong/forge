# FORGE Usage Guide

Framework for Organic-molecule Generation via Reflection-Guided Evaluation
## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  LLM Provider (OpenAI-compatible API)                               │
│  └─ Calls LLM to generate SMILES                                    │
├──────────────────────────────────────────────────────────────────────┤
│  LLMCore / SimpleLLMCore — Main Orchestrator                        │
│  Loop: RAG → Prompt → LLM → Evaluator → Record → Feedback → RAG    │
├──────────────────────────────────────────────────────────────────────┤
│  Evaluator / SimpleEvaluator — Molecular Property Evaluator         │
│  Evaluator:      XTB(all) → DFT(top N)                             │
│  SimpleEvaluator: XTB(all), no DFT                                  │
├──────────────────────────────────────────────────────────────────────┤
│  ReflectionRAG — Molecular Retrieval + Incremental Index            │
│  Base: QM9 ~124K molecules (FAISS binary fingerprints)             │
│  Fork mechanism: each experiment gets an isolated copy             │
├──────────────────────────────────────────────────────────────────────┤
│  ResultStore — Result Persistence                                   │
│  round_N/prompt.json, llm_response.txt, usage.json,                │
│  eval_results.parquet, feedback_table.csv, summary.json             │
└──────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. `llm_core` / `llm_core_simple` — Main Orchestrator

```python
from llm_core import LLMCore                    # DFT version
from llm_core_simple import SimpleLLMCore       # XTB-only version
```

Both share the same interface:

```python
core = LLMCore(
    provider=provider,           # LLMProvider
    record_prompts=True,         # Save prompts to disk
    reflection_mode="spr",       # Feedback mode: "spr" / "scalar"
    use_rag=True,                # Enable RAG retrieval
)

result = core.run(
    target={"HOMO-LUMO gap": "3.0 eV"},  # Target property
    n_rounds=10,                          # Iteration rounds
    n_per_round=20,                       # Molecules per round
    n_feedback=5,                         # Feedback molecules per round
    evaluator=evaluator,                  # Evaluator / SimpleEvaluator
    rag=rag,                              # RAGEngine instance
    run_name="my_experiment",             # Experiment name
    runs_dir="runs",                      # Output root directory
    seed=42,                              # Random seed
    max_tokens=16384,                     # LLM max_tokens
    temperature=0.8,                      # LLM temperature
    gap_range_margin=2.0,                 # RAG gap search range (±eV)
)
```

Return value:
```python
{
    "run_dir": "runs/my_experiment",       # Output directory
    "best": {"smiles": "...", "dft_gap_eV": 3.0012, ...},  # Best molecule
    "rounds": [...],                       # Per-round summary
    "failed": False,                       # Whether all rounds failed
}
```

### 2. `evaluator` — DFT Property Evaluator

```python
from evaluator import Evaluator

evaluator = Evaluator(
    n_parallel=4,               # Parallelism
    use_correlation=True,       # XTB→DFT correlation calibration
    xtb_bin="/path/to/xtb",     # XTB binary path
    dft_functional="B3LYP",     # DFT functional
    dft_basis="6-31G(2d,p)",    # DFT basis set
)
df = evaluator.evaluate(
    smiles_list=["CCO", "c1ccccc1"],
    target_gap=3.0,
    n_top=3,                    # Top N molecules for DFT
)
# df columns: smiles, xtb_gap_eV, dft_gap_eV, dft_homo_eV,
#             dft_lumo_eV, dft_dipole_D, dft_energy_Ha, ...
```

Pipeline:
1. Run XTB geometry optimization on all n molecules
2. Optional: MLP calibration (XTB → DFT gap)
3. Sort by |gap - target|, pick top N
4. Run pySCF DFT on top N molecules
5. Return DataFrame

### 3. `simple_evaluator` — XTB-Only Evaluator

```python
from simple_evaluator import SimpleEvaluator

evaluator = SimpleEvaluator(
    n_parallel=4,               # Parallelism
    xtb_bin="/path/to/xtb",     # XTB binary path
)
df = evaluator.evaluate(
    smiles_list=["CCO", "c1ccccc1"],
    target_gap=3.0,
    n_top=3,
)
# df columns: smiles, xtb_gap_eV, xtb_homo_eV, xtb_lumo_eV,
#             xtb_dipole_D, xtb_energy_Ha, ...
```

Pipeline:
1. Run XTB geometry optimization (with --dipole) on all n molecules
2. Sort by |xtb_gap - target|, pick top N
3. Return DataFrame

No DFT, no correlation calibration.

### 4. `reflection_rag` — Retrieval-Augmented Generation

```python
from reflection_rag import RAGEngine

# One-time base build (skips if exists)
RAGEngine.build_base()          # Load QM9 ~124K molecules

# Fork an isolated instance per experiment
rag = RAGEngine.fork()          # Copy-on-write, does not pollute base

# Query similar molecules
results = rag.query(
    feedbacks=[...],            # Historical feedback
    n_samples=20,               # Desired sample count
    gap_range_margin=2.0,       # Gap search range
)
# results: [{smiles, xtb_gap, ...}, ...]

# Update feedback index (after each round)
rag.add_reflection(
    smiles_list=["..."],        # Successfully evaluated molecules
    properties_list=[{...}],    # Corresponding property dicts
)
```

### 5. `llm_core.providers` — LLM Provider

```python
from llm_core.providers import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-v4-flash",
    timeout=300,                # HTTP timeout (seconds)
)
```

Built-in retry: 3 attempts with exponential backoff (4s, 8s) for 429/503/timeout.

### 6. `experiment.models_config` — Model Configuration Loader

```python
from experiment.models_config import load_models, run_dir_name

models = load_models()          # Load from api_llm.txt
cfg = models[0]                 # ModelConfig(model, api_key, base_url, provider, timeout)

name = run_dir_name(cfg)        # → "DeepSeek_v4-flash"
```

## Writing a run.py

### Standard Mode (with DFT)

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

### Simple Mode (XTB only)

```python
import logging, sys, os
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

# Only two imports differ
from llm_core_simple import SimpleLLMCore    # ← different
from simple_evaluator import SimpleEvaluator # ← different
from llm_core.providers import OpenAICompatibleProvider
from reflection_rag import RAGEngine
from experiment.models_config import load_models

cfg = [m for m in load_models() if m.model == "deepseek-v4-flash"][0]
provider = OpenAICompatibleProvider(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model)

evaluator = SimpleEvaluator(n_parallel=4)    # ← XTB only, no DFT
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

## Standard vs Simple Mode

| Aspect | Standard (llm_core + Evaluator) | Simple (llm_core_simple + SimpleEvaluator) |
|---|---|---|
| **Pipeline** | XTB(all) → DFT(top N) | XTB(all), no DFT |
| **Feedback fields** | DFT gap/HOMO/LUMO/dipole/energy | **XTB** gap/HOMO/LUMO/dipole/energy |
| **Field prefix** | `dft_gap_eV`, `dft_homo_eV`, ... | `xtb_gap_eV`, `xtb_homo_eV`, ... |
| **Speed** | Slow (DFT: 1-5 min per molecule) | Fast (XTB: <1s per molecule) |
| **Compute cost** | High (DFT resources) | XTB only |
| **Correlation** | Optional (MLP calibration) | None |
| **Feedback latency** | High (waiting for DFT) | Low (XTB is instant) |
| **Accuracy** | DFT values (close to ground truth) | XTB values (approximate, same trend) |
| **Use case** | Final validation needing DFT accuracy | Fast iteration, gap sweeps, method comparison |

### Field Mapping in SimpleLLMCore

`SimpleLLMCore` automatically maps `xtb_*` to `dft_*` for downstream compatibility:

```
xtb_gap_eV      → dft_gap_eV
xtb_homo_eV     → dft_homo_eV
xtb_lumo_eV     → dft_lumo_eV
xtb_dipole_D    → dft_dipole_D
xtb_energy_Ha   → dft_energy_Ha
xtb_success     → dft_success
```

This means results from `SimpleLLMCore` have the same directory structure as the standard version and can be analyzed with the same tools.

## Result Directory Structure

```
runs/my_experiment/
├── manifest.json              # Overview (config, round summaries)
├── summary.json               # Best molecule + aggregate stats
├── feedback_table.csv         # Cumulative feedback table
├── round_01/
│   ├── prompt.json            # Full prompt sent to LLM
│   ├── llm_response.txt       # Raw LLM response
│   ├── usage.json             # Token usage
│   ├── generated_smiles.txt   # Parsed SMILES list
│   ├── eval_results.parquet   # Evaluation results (DataFrame)
│   └── rag_context.json       # RAG-retrieved reference molecules
├── round_02/ ...
└── round_10/ ...
```

## Model Configuration (api_llm.txt)

Format: `model_name, api_key, base_url`

```
deepseek-v4-flash,sk-xxx,https://api.deepseek.com/v1
kimi-k2.6,sk-xxx,https://api.moonshot.cn/v1
glm-5.1,sk-xxx,https://api.glm.cn/v4
...
```

Note: `kimi-k2.6` requires `temperature=1.0` (preconfigured in `MODEL_TEMPERATURE`).

## Quick Reference

```bash
# Standard DFT experiment
python experiment/run.py

# Multi-model comparison
python experiment/run_compare.py

# Single-model gap sweep (XTB)
python experiment/run_gap_sweep.py --model deepseek-v4-flash

# Per-molecule mode (dense reflection)
python experiment/run_permolecule.py

# Result analysis
python experiment/summarize.py
```
