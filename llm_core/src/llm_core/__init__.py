"""FORGE LLM Core — LLM 驱动的分子设计主编排器.

固定轮次迭代: RAG 检索 → Prompt 组装 → LLM 生成 → Evaluator 评估 → 记录.

用法:
    from llm_core import LLMCore, ResultStore
    from llm_core.providers import DeepSeekProvider
    from evaluator import Evaluator
    from reflection_rag import RAGEngine

    core = LLMCore(provider=DeepSeekProvider(api_key="sk-..."))
    result = core.run(
        target={"HOMO-LUMO gap": "5.0 eV"},
        n_rounds=5, n_per_round=20,
        evaluator=Evaluator(),
        rag=RAGEngine.fork(),
        run_name="exp_001",
    )

    # 事后读取反馈表格
    df = ResultStore.read_feedback_table("runs/exp_001", target_gap=5.0)
"""

from .engine import LLMCore
from .result_store import ResultStore

__all__ = ["LLMCore", "ResultStore"]
