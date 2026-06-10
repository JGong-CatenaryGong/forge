"""LLM 模型配置 — 从 api_llm.txt 解析所有待测模型.

api_llm.txt 格式（逗号分隔）:
    <model_name>,<api_key>,<base_url>

用法:
    from experiment.models_config import load_models
    models = load_models()
    for m in models:
        print(m.model, m.base_url)
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    model: str
    api_key: str
    base_url: str
    provider: str
    timeout: float = 300.0

_PROVIDER_PREFIXES = {
    "deepseek": "DeepSeek",
    "kimi": "Moonshot",
    "glm": "Zhipu",
    "minimax": "MiniMax",
    "qwen": "Qwen",
}


def _infer_provider(model: str) -> str:
    lower = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if lower.startswith(prefix):
            return provider
    return model.split("-")[0].capitalize()


def load_models(api_file: str = None) -> List[ModelConfig]:
    """从 api_llm.txt 加载所有模型配置.

    Args:
        api_file: 配置文件路径。默认为项目根目录下的 api_llm.txt。

    Returns:
        ModelConfig 列表，按文件顺序排列。
    """
    if api_file is None:
        exp_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(exp_dir)
        api_file = os.path.join(project_root, "api_llm.txt")

    models = []
    with open(api_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            model_name, api_key, base_url = parts[0], parts[1], parts[2]
            provider = _infer_provider(model_name)
            models.append(ModelConfig(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
            ))

    return models


def run_dir_name(cfg: ModelConfig) -> str:
    short = cfg.model
    lower = short.lower()
    for prefix in _PROVIDER_PREFIXES:
        if lower.startswith(prefix) and len(lower) > len(prefix):
            sep = lower[len(prefix)]
            if not sep.isalpha():
                short = short[len(prefix):].lstrip(".-")
                break
    return f"{cfg.provider}_{short}"
