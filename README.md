# FORGE

**Framework for Organic-molecule Generation via Reflection-Guided Evaluation**

FORGE is an iterative molecular design system that uses large language models
to generate SMILES molecules targeting specific electronic properties (e.g.,
HOMO-LUMO gap). Each cycle: retrieves similar molecules from QM9 via RAG →
prompts an LLM → evaluates candidates with quantum chemistry (XTB ± DFT) →
feeds results back to the LLM for refinement.

## Quick Start

```bash
conda create -n forge python=3.12
conda activate forge
pip install rdkit pandas numpy scikit-learn openai pyscf
pip install faiss-cpu pyarrow joblib

# Install local packages
pip install -e llm_core
pip install -e evaluator
pip install -e reflection_rag
# Optional: XTB-only mode (no DFT)
pip install -e simple_evaluator
pip install -e llm_core_simple

# XTB binary (https://github.com/grimme-lab/xtb)
export XTB_BIN=/path/to/xtb
```

Configure API keys in `api_llm.txt`:
```
deepseek-v4-pro,sk-xxx,https://api.deepseek.com/v1
```

## Packages

| Package | Role | DFT |
|---|---|---|
| `llm_core` | Main orchestrator (iterative loop) | ✅ |
| `evaluator` | XTB + pySCF property calculator | ✅ |
| `reflection_rag` | FAISS molecular retrieval (QM9) | — |
| `simple_evaluator` | XTB-only evaluator (no DFT) | ❌ |
| `llm_core_simple` | XTB-only orchestrator | ❌ |

## Benchmark (DFT batch — target gap 3.0 eV)

7 models × 3 seeds × 10 rounds × 20 candidates, sorted by best |dev| across seeds.

| Model | Best \|dev\| | Med \|dev\| | Mean ± SE |
|---|---|---|---|
| **DeepSeek v4-pro** | **0.0089** | 0.0473 | 0.0445 ± 0.0198 |
| Mimo v2.5-pro | 0.0053 | 0.0097 | 0.0250 ± 0.0176 |
| Qwen 3.7-max | 0.0024 | 0.0112 | 0.0331 ± 0.0265 |
| DeepSeek v4-flash | 0.0056 | 0.0137 | 0.0567 ± 0.0471 |
| GLM 5.1 | 0.0030 | 0.0140 | 0.0315 ± 0.0232 |
| MiniMax M3 | 0.0103 | 0.0141 | 0.0141 ± 0.0038 |
| Doubao | 0.0004 | 0.0488 | 0.0509 ± 0.0298 |

## License

MIT
