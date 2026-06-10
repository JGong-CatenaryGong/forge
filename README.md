# MolAgent

**LLM-driven molecular design with quantum chemistry feedback.**

MolAgent is an iterative molecular design system that uses large language models
to generate SMILES molecules targeting specific electronic properties (e.g.,
HOMO-LUMO gap). Each iteration cycle: retrieves similar molecules from QM9 via
RAG → prompts an LLM → evaluates generated molecules with quantum chemistry
(XTB + optional DFT) → feeds results back to the LLM for the next round.

## How It Works

```
                  ┌──────────┐
                  │   QM9    │
                  │  ~124K   │
                  │ molecules│
                  └────┬─────┘
                       │ RAG
                       ▼
   ┌─────────────────────────────────┐
   │  LLM + Prompt + Feedback       │
   │  → generates SMILES candidates │
   └────────────┬────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │  XTB Geometry Optimization      │
   │  + optional DFT (pySCF)         │
   │  → gap, HOMO, LUMO, dipole     │
   └────────────┬────────────────────┘
                │
         ┌──────┴──────┐
         ▼              ▼
   ┌──────────┐   ┌──────────┐
   │ Feedback │   │  Record  │
   │ to LLM   │   │  to disk │
   └──────────┘   └──────────┘
```

## Quick Start

```bash
# Install dependencies (conda environment recommended)
conda create -n molagent python=3.12
conda activate molagent
pip install rdkit pandas numpy scikit-learn openai pyscf
pip install faiss-cpu pyarrow joblib

# Install local packages
pip install -e llm_core
pip install -e evaluator
pip install -e reflection_rag

# Optional: XTB-only mode (no DFT)
pip install -e simple_evaluator
pip install -e llm_core_simple

# Install xtb (https://github.com/grimme-lab/xtb)
export XTB_BIN=/path/to/xtb
```

Then configure your API keys in `api_llm.txt`:

```
deepseek-v4-flash,sk-xxx,https://api.deepseek.com/v1
```

### Run an experiment

```bash
# Standard DFT mode
python experiment/run.py

# Multi-model comparison
python experiment/run_compare.py

# XTB-only gap sweep (no DFT)
python experiment/run_gap_sweep.py --model deepseek-v4-flash

# Analyze results
python experiment/summarize.py
```

## Packages

| Package | Role | DFT |
|---|---|---|
| `llm_core` | Main orchestrator (iterative design loop) | ✅ |
| `evaluator` | XTB + pySCF property calculator | ✅ |
| `reflection_rag` | FAISS-based molecular retrieval (QM9) | — |
| `experiment/` | Experiment scripts & model config | — |
| `simple_evaluator` | XTB-only evaluator (no DFT) | ❌ |
| `llm_core_simple` | XTB-only orchestrator | ❌ |

Both modes share the same interfaces, prompts, RAG engine, and result storage format.

## Key Features

- **Iterative molecular design**: 10 rounds × 20 candidates per round
- **Dual evaluator modes**: DFT-accurate or XTB-fast
- **RAG retrieval**: FAISS binary fingerprint index over QM9 (124K molecules)
- **Reflection feedback**: SPR mode highlights gap | score mode gives scalar breakdown
- **Multi-model support**: test any OpenAI-compatible LLM
- **Real-time persistence**: per-round prompts, responses, and evaluations saved to disk

## Experiment Results

See [experiment/runs/compare/](experiment/runs/compare/) for benchmark data (target gap: 3.0 eV, 8 models × 3 seeds each).

Top performers (median |deviation from target| across 3 seeds):

| Rank | Model | Med \|dev\| | Mean ± SE |
|---|---|---|---|
| 1 | Mimo v2.5-pro | **0.0097** | 0.0250 ± 0.0176 |
| 2 | Qwen 3.7-max | 0.0112 | 0.0331 ± 0.0265 |
| 3 | DeepSeek v4-flash | 0.0137 | 0.0567 ± 0.0471 |
| 4 | GLM 5.1 | 0.0140 | 0.0315 ± 0.0232 |
| 5 | DeepSeek v4-pro | 0.0473 | 0.0445 ± 0.0198 |

Full report: [experiment/benchmark_report.typ](experiment/benchmark_report.typ)

## License

MIT
