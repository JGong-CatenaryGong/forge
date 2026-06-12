"""FORGE Spectra LLM Core — LLM 驱动分子设计 (目标: 吸收光谱颜色).

使用 SpectraEvaluator, 目标为 CIE 1931 感知色 (补色) 坐标.
"""
from .core import SpectraLLMCore

__all__ = ["SpectraLLMCore"]
