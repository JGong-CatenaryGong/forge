"""LLM Provider 抽象层 — 统一接口 + DeepSeek/OpenAI 实现.

使用 OpenAI 兼容的 chat completions API，通过参数切换后端.

用法:
    from llm_core.providers import DeepSeekProvider

    provider = DeepSeekProvider(api_key="sk-...", model="deepseek-chat")
    response = provider.generate([{"role": "user", "content": "Hello"}])
    # → (text, usage_dict)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """LLM 后端抽象基类."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """生成回复.

        Returns:
            (response_text, usage_dict).
            usage_dict: {"prompt_tokens": N, "completion_tokens": N,
                         "total_tokens": N, "time_s": N}.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容 API 通用实现.

    支持 DeepSeek, OpenAI, 以及任何兼容 /v1/chat/completions 的服务.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> Tuple[str, Dict]:
        last_error = None
        for attempt in range(3):
            t0 = time.monotonic()
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                elapsed = time.monotonic() - t0
                choice = resp.choices[0]
                text = choice.message.content or ""
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    "time_s": round(elapsed, 2),
                    "finish_reason": choice.finish_reason,
                }
                return text, usage
            except Exception as e:
                elapsed = time.monotonic() - t0
                is_retryable = any(
                    s in str(e).lower()
                    for s in ["429", "503", "overloaded", "rate_limit", "try again later",
                              "internal_server_error", "service_unavailable"]
                )
                if is_retryable and attempt < 2:
                    wait = 2 ** (attempt + 2)  # 4s, 8s
                    logger.warning(f"LLM API transient error ({self._model}), "
                                   f"retry in {wait}s (attempt {attempt+1}/3): {e}")
                    time.sleep(wait)
                    last_error = e
                else:
                    logger.error(f"LLM API error ({self._model}): {e}")
                    raise


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        timeout: float = 120.0,
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
            timeout=timeout,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        timeout: float = 120.0,
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            timeout=timeout,
        )
