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

Configure API keys (copy template, then edit):
```
cp api_llm.example.txt api_llm.txt
# 编辑 api_llm.txt, 填入你的真实 key
```

## Packages

| Package | Role | DFT |
|---|---|---|
| `llm_core` | Main orchestrator (iterative loop) | ✅ |
| `evaluator` | XTB + pySCF property calculator | ✅ |
| `reflection_rag` | FAISS molecular retrieval (QM9) | — |
| `simple_evaluator` | XTB-only evaluator (no DFT) | ❌ |
| `llm_core_simple` | XTB-only orchestrator | ❌ |

## License

MIT
