"""SpectraLLMCore — LLM 分子设计编排器 (目标: CIE 1931 感知色).

迭代循环: LLM 生成 → SpectraEvaluator 评估 → 反馈 → 记录.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from llm_core.providers import LLMProvider
from llm_core.result_store import ResultStore
from .prompts import build_messages

logger = logging.getLogger(__name__)


def _build_feedback(eval_df: pd.DataFrame, round_id: int, n_feedback: int) -> Dict:
    """构建反馈字典."""
    if "td_dft_success" in eval_df.columns:
        success = eval_df[eval_df["td_dft_success"] == True]
    else:
        success = eval_df
    molecules = success.head(n_feedback).to_dict("records") if len(success) > 0 else []
    return {
        "round": round_id,
        "n_total": len(eval_df),
        "n_success": len(success),
        "molecules": molecules,
    }


class SpectraLLMCore:
    """LLM 分子设计编排器 (目标: CIE 1931 感知色).

    用法:
        core = SpectraLLMCore(provider=provider, evaluator=spectra_evaluator)
        result = core.run(
            target_cie=(0.3, 0.6),
            n_rounds=10, n_per_round=20, n_feedback=5,
        )
    """

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Any,
        record_prompts: bool = True,
    ):
        self.provider = provider
        self.evaluator = evaluator
        self.record_prompts = record_prompts

    def run(
        self,
        target_cie: Tuple[float, float],
        n_rounds: int = 10,
        n_per_round: int = 20,
        n_feedback: int = 5,
        run_name: str = "default",
        runs_dir: str = "runs",
        seed: int = 42,
        max_tokens: int = 16384,
        temperature: float = 0.8,
    ) -> Dict[str, Any]:
        """运行固定轮次的分子设计迭代 (目标: 感知色 CIE).

        Args:
            target_cie: 目标感知色 (补色) CIE 1931 (x, y).
            n_rounds: 迭代轮数.
            n_per_round: 每轮生成分子数.
            n_feedback: 每轮反馈分子数.
            run_name: 运行名称.
            runs_dir: 结果根目录.
            seed: 随机种子.
            max_tokens: LLM max_tokens.
            temperature: LLM 温度.

        Returns:
            Dict with keys: run_dir, best, rounds, failed.
        """
        store = ResultStore(
            run_dir=os.path.join(runs_dir, run_name),
            record_prompts=self.record_prompts,
        )
        all_round_feedbacks: List[Dict] = []
        global_best = None
        global_best_cie_dev = float("inf")

        t_run_start = time.monotonic()
        target_x, target_y = target_cie
        logger.info(f"=== SpectraLLMCore run: {run_name} ===")
        logger.info(f"  target CIE=({target_x:.4f}, {target_y:.4f}), "
                     f"rounds={n_rounds}, per_round={n_per_round}, seed={seed}")

        for round_id in range(1, n_rounds + 1):
            t_round_start = time.monotonic()
            logger.info(f"  Round {round_id}/{n_rounds}")

            # ── 1. 组装 Prompt ──
            messages = build_messages(
                target_cie=target_cie,
                round_feedbacks=all_round_feedbacks if all_round_feedbacks else None,
                n_per_round=n_per_round,
            )

            # ── 2. LLM 生成 ──
            t_llm_start = time.monotonic()
            raw_response, usage = self.provider.generate(
                messages, max_tokens=max_tokens, temperature=temperature,
            )
            t_llm = time.monotonic() - t_llm_start

            # ── 3. 解析 SMILES ──
            smiles_list = _parse_smiles(raw_response, n_per_round)
            logger.info(f"  Generated {len(smiles_list)} SMILES ({usage['total_tokens']} tokens, {t_llm:.1f}s)")

            # ── 4. SpectraEvaluator 评估 ──
            if smiles_list:
                try:
                    eval_df = self.evaluator.evaluate(
                        smiles_list=smiles_list,
                        target_cie=target_cie,
                        n_top=n_feedback,
                    )
                except Exception as e:
                    logger.error(f"  SpectraEvaluator error: {e}")
                    eval_df = pd.DataFrame({"smiles": smiles_list, "td_dft_success": [False] * len(smiles_list)})
            else:
                logger.warning(f"  Round {round_id}: 0 SMILES parsed, skipping evaluation")
                eval_df = pd.DataFrame({"smiles": [], "td_dft_success": []})

            # ── 5. 提取成功子集 ──
            success_col = "td_dft_success"
            eval_success = eval_df[eval_df[success_col] == True] if success_col in eval_df.columns else pd.DataFrame()

            n_success = len(eval_success)
            feedback = _build_feedback(eval_df, round_id, n_feedback)

            # ── 6. 记录最佳 ──
            if n_success > 0:
                for _, row in eval_success.iterrows():
                    dev = row.get("cie_deviation")
                    if dev is not None and dev < global_best_cie_dev:
                        global_best_cie_dev = dev
                        global_best = row.to_dict()

            # ── 7. 落盘 ──
            store.save_round(
                round_id=round_id,
                prompt_messages=messages,
                llm_response=raw_response,
                llm_usage=usage,
                eval_df=eval_df,
                smiles_generated=smiles_list,
            )
            # 保存光谱专用 CSV
            if "cie_x" in eval_df.columns:
                spec_cols = [c for c in ["smiles", "cie_x", "cie_y", "lambda_max_nm",
                                         "dft_gap_eV", "cie_deviation", "nstates"] if c in eval_df.columns]
                csv_path = os.path.join(store.run_dir, f"round_{round_id:02d}", "spectral_data.csv")
                eval_df[spec_cols].to_csv(csv_path, index=False, float_format="%.4f")

            all_round_feedbacks.append(feedback)
            t_round = time.monotonic() - t_round_start
            logger.info(f"  Round {round_id} done: {n_success}/{len(smiles_list)} TD-DFT ok ({t_round:.1f}s)")

        t_total = time.monotonic() - t_run_start
        config = {
            "provider": self.provider.model_name,
            "target_cie_x": target_x,
            "target_cie_y": target_y,
            "n_rounds": n_rounds,
            "n_per_round": n_per_round,
            "n_feedback": n_feedback,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "run_name": run_name,
            "total_wall_time_s": round(t_total, 1),
        }
        store.save_summary(config, {"best_molecule": global_best})

        logger.info(f"=== SpectraLLMCore run complete: {store.run_dir} ===")
        logger.info(f"  total: {t_total:.0f}s, best CIE dev: {global_best_cie_dev:.6f}")

        return {
            "run_dir": store.run_dir,
            "best": global_best,
            "rounds": store._rounds,
            "failed": global_best is None,
        }


# ── SMILES 解析 ──
import re
_SMILES_LINE = re.compile(r"^\s*(?:\d+[\.\)]\s*)?(\S+)\s*$")

def _parse_smiles(raw_response: str, expected_count: int) -> List[str]:
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    smiles = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^[A-Za-z]+$", line) and len(line) > 10:
            continue
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
