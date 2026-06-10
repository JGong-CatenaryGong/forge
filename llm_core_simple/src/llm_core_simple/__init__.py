"""LLMCore Simple — LLM 驱动分子设计 (纯 XTB，无 DFT).

复用原 llm_core 的 providers/prompts/result_store，
用 SimpleEvaluator 替代原 Evaluator（只跑 XTB，无 DFT）。

用法:
    from llm_core_simple import SimpleLLMCore
    from simple_evaluator import SimpleEvaluator
    from llm_core.providers import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="...", base_url="...", model="...")
    evaluator = SimpleEvaluator(n_parallel=4)
    core = SimpleLLMCore(provider=provider, evaluator=evaluator)

    result = core.run(
        target={"HOMO-LUMO gap": "3.0 eV"},
        n_rounds=20,
        n_per_round=1,
        n_feedback=1,
        seed=42,
    )
"""

from .simple_llm_core import SimpleLLMCore

__all__ = ["SimpleLLMCore"]
