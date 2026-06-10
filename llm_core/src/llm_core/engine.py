"""LLMCore — LLM 驱动的分子设计主编排器.

固定轮次迭代循环: RAG 检索 → Prompt 组装 → LLM 生成 → Evaluator 评估 → 记录 → 更新 RAG.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from .providers import LLMProvider
from .prompts import build_messages
from .result_store import ResultStore

if TYPE_CHECKING:
    from evaluator import Evaluator
    from reflection_rag import RAGEngine

logger = logging.getLogger(__name__)

_SMILES_LINE = re.compile(r"^\s*(?:\d+[\.\)]\s*)?(\S+)\s*$")


class LLMCore:
    """LLM 分子设计主编排器."""

    def __init__(self, provider: LLMProvider, record_prompts: bool = True,
                 reflection_mode: str = "spr", use_rag: bool = True):
        self.provider = provider
        self.record_prompts = record_prompts
        self.reflection_mode = reflection_mode
        self.use_rag = use_rag
    def run(
        self,
        target: Dict[str, str],
        n_rounds: int,
        n_per_round: int,
        evaluator,
        rag,
        n_feedback: int = 5,
        run_name: str = "default",
        runs_dir: str = "runs",
        seed: int = 42,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        gap_range_margin: float = 1.5,
        **provider_kwargs,
    ) -> Dict[str, Any]:
        """运行固定轮次的分子设计迭代."""
        import numpy as np
        np.random.seed(seed)

        target_gap = self._parse_target_gap(target)

        store = ResultStore(
            run_dir=os.path.join(runs_dir, run_name),
            record_prompts=self.record_prompts,
        )

        global_best = None
        global_best_gap_diff = float("inf")
        all_round_feedbacks: List[Dict] = []

        logger.info(f"=== LLMCore run start: {run_name} ===")
        logger.info(f"  target={target}, rounds={n_rounds}, per_round={n_per_round}, "
                     f"provider={self.provider.model_name}, reflection={self.reflection_mode}")

        t_run_start = time.monotonic()

        for round_id in range(1, n_rounds + 1):
            t_round_start = time.monotonic()
            logger.info(f"--- Round {round_id}/{n_rounds} ---")

            # 1. RAG 检索 (可关闭)
            if self.use_rag:
                gap_min, gap_max = None, None
                if target_gap is not None:
                    gap_min = max(0, target_gap - gap_range_margin)
                    gap_max = target_gap + gap_range_margin
                rag_results = rag.query(
                    gap_range=(gap_min, gap_max) if gap_min is not None else None,
                    top_k=50,
                )
            else:
                rag_results = []

            # 2. 组装 prompt
            messages = build_messages(
                task="generate",
                target=target,
                rag_results=rag_results,
                round_feedbacks=all_round_feedbacks if all_round_feedbacks else None,
                n_per_round=n_per_round,
                system_prompt=system_prompt,
                reflection_mode=self.reflection_mode,
            )

            # 3. LLM 生成
            t_llm_start = time.monotonic()
            raw_response, usage = self.provider.generate(
                messages, max_tokens=max_tokens, temperature=temperature, **provider_kwargs,
            )
            t_llm = time.monotonic() - t_llm_start

            # 4. 解析 SMILES
            smiles_list = self._parse_smiles(raw_response, n_per_round)
            logger.info(f"  Generated {len(smiles_list)} SMILES ({usage['total_tokens']} tokens, {t_llm:.1f}s)")

            # 5. Evaluator 评估
            if smiles_list:
                try:
                    eval_df = evaluator.evaluate(
                        smiles_list=smiles_list,
                        target_gap=target_gap if target_gap is not None else 0.0,
                        n_top=n_feedback,
                    )
                except Exception as e:
                    logger.error(f"  Evaluator error: {e}")
                    eval_df = pd.DataFrame({"smiles": smiles_list, "dft_success": [False] * len(smiles_list)})
            else:
                eval_df = pd.DataFrame({"smiles": [], "dft_success": []})

            # 6. 提取成功子集
            dft_col = "dft_success"
            if dft_col in eval_df.columns:
                eval_success = eval_df[eval_df[dft_col] == True]
            else:
                eval_success = pd.DataFrame()

            n_success = len(eval_success)
            feedback = self._build_feedback(eval_df, round_id, n_feedback)

            # 7. 更新 RAG
            if self.use_rag and n_success > 0:
                top_smiles = eval_success["smiles"].head(n_feedback).tolist()
                top_props = eval_success.head(n_feedback).to_dict("records")
                rag.add_reflection(top_smiles, top_props)
            if n_success > 0 and target_gap is not None:
                for _, row in eval_success.iterrows():
                    dft_gap = row.get("dft_gap_eV")
                    if dft_gap is None:
                        continue
                    diff = abs(dft_gap - target_gap)
                    if diff < global_best_gap_diff:
                        global_best_gap_diff = diff
                        global_best = row.to_dict()

            # 9. 记录落盘
            store.save_round(
                round_id=round_id,
                prompt_messages=messages,
                llm_response=raw_response,
                llm_usage=usage,
                eval_df=eval_df,
                rag_context=rag_results,
                smiles_generated=smiles_list,
            )

            all_round_feedbacks.append(feedback)
            t_round = time.monotonic() - t_round_start
            logger.info(f"  Round {round_id} done: {n_success}/{len(smiles_list)} passed DFT ({t_round:.1f}s)")

        # 汇总
        t_total = time.monotonic() - t_run_start
        config = {
            "provider": self.provider.model_name,
            "target": target,
            "n_rounds": n_rounds,
            "n_per_round": n_per_round,
            "n_feedback": n_feedback,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "gap_range_margin": gap_range_margin,
            "reflection_mode": self.reflection_mode,
            "run_name": run_name,
            "total_wall_time_s": round(t_total, 1),
        }
        store.save_summary(config, {"best_molecule": global_best})

        logger.info(f"=== LLMCore run complete: {store.run_dir} ===")
        logger.info(f"  total: {t_total:.0f}s, best gap diff: {global_best_gap_diff:.4f}")

        return {
            "run_dir": store.run_dir,
            "best": global_best,
            "rounds": store._rounds,
            "failed": global_best is None,
        }

    @staticmethod
    def _parse_target_gap(target: Dict[str, str]) -> Optional[float]:
        for val in target.values():
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                m = re.search(r"([\d.]+)", val)
                if m:
                    return float(m.group(1))
        return None

    @staticmethod
    def _parse_smiles(raw_response: str, expected_count: int) -> List[str]:
        # Strip <think>...</think> reasoning blocks
        text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
        # Strip markdown code fences
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        smiles = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip lines that are plain English words (all letters, hyphens, no digits)
            if re.match(r"^[A-Za-z\s\-]+$", line):
                continue
            # Skip XML/HTML tags
            if "<" in line or ">" in line:
                continue
            m = _SMILES_LINE.match(line)
            if m:
                token = m.group(1)
                if any(c.isalpha() for c in token):
                    smiles.append(token)
                    if len(smiles) >= expected_count:
                        break
        return smiles

    @staticmethod
    def _build_feedback(eval_df: pd.DataFrame, round_id: int, n_feedback: int) -> Dict:
        if "dft_success" in eval_df.columns:
            success = eval_df[eval_df["dft_success"] == True]
        else:
            success = eval_df
        molecules = success.head(n_feedback).to_dict("records") if len(success) > 0 else []
        return {
            "round": round_id,
            "n_total": len(eval_df),
            "n_success": len(success),
            "molecules": molecules,
        }
