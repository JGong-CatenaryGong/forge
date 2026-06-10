"""SimpleLLMCore — LLM 驱动分子设计主编排器 (纯 XTB 版).

迭代循环: RAG 检索 → Prompt 组装 → LLM 生成 → SimpleEvaluator 评估 → 记录 → 更新 RAG.

与原 LLMCore 的区别:
    - 使用 SimpleEvaluator (纯 XTB, 无 DFT)
    - 反馈字段映射: xtb_* → dft_* (使 prompts/result_store 兼容)
    - 无相关性校准
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from llm_core.providers import LLMProvider
from llm_core.prompts import build_messages
from llm_core.result_store import ResultStore

if TYPE_CHECKING:
    from simple_evaluator import SimpleEvaluator
    from reflection_rag import RAGEngine

logger = logging.getLogger(__name__)

_SMILES_LINE = re.compile(r"^\s*(?:\d+[\.\)]\s*)?(\S+)\s*$")


# ── 字段映射 ────────────────────────────────────────────────

def _map_xtb_to_dft(df: pd.DataFrame) -> pd.DataFrame:
    """将 SimpleEvaluator 的 xtb_* 列名映射为 dft_* 列名，兼容原 pipeline."""
    mapping = {
        "xtb_gap_eV": "dft_gap_eV",
        "xtb_homo_eV": "dft_homo_eV",
        "xtb_lumo_eV": "dft_lumo_eV",
        "xtb_dipole_D": "dft_dipole_D",
        "xtb_energy_Ha": "dft_energy_Ha",
        "xtb_success": "dft_success",
        "xtb_deviation_eV": "dft_deviation_eV",
    }
    return df.rename(columns=mapping)


# ── SMILES 解析 ─────────────────────────────────────────────

def _parse_smiles(raw_response: str, expected_count: int) -> List[str]:
    """解析 LLM 输出中的 SMILES."""
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    smiles = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip English words (all letters only, >10 chars)
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


# ── 目标解析 ────────────────────────────────────────────────

def _parse_target_gap(target: Dict[str, str]) -> Optional[float]:
    for val in target.values():
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            m = re.search(r"([\d.]+)", val)
            if m:
                return float(m.group(1))
    return None


# ── 反馈构建 ────────────────────────────────────────────────

def _build_feedback(eval_df: pd.DataFrame, round_id: int, n_feedback: int) -> Dict:
    success_col = "dft_success"
    if success_col in eval_df.columns:
        success = eval_df[eval_df[success_col] == True]
    else:
        success = eval_df
    molecules = success.head(n_feedback).to_dict("records") if len(success) > 0 else []
    return {
        "round": round_id,
        "n_total": len(eval_df),
        "n_success": len(success),
        "molecules": molecules,
    }


# ── 主编排器 ────────────────────────────────────────────────

class SimpleLLMCore:
    """LLM 分子设计主编排器 (纯 XTB 版).

    与原 LLMCore 相同的接口，但使用 SimpleEvaluator 替代 Evaluator。
    """

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: "SimpleEvaluator",
        record_prompts: bool = True,
        reflection_mode: str = "spr",
        use_rag: bool = True,
    ):
        self.provider = provider
        self.evaluator = evaluator
        self.record_prompts = record_prompts
        self.reflection_mode = reflection_mode
        self.use_rag = use_rag

    def run(
        self,
        target: Dict[str, str],
        n_rounds: int,
        n_per_round: int,
        n_feedback: int,
        evaluator: Any = None,  # compat
        rag: Optional["RAGEngine"] = None,
        run_name: str = "run",
        runs_dir: str = "runs",
        seed: Optional[int] = None,
        max_tokens: int = 16384,
        temperature: float = 0.8,
        gap_range_margin: float = 2.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """运行固定轮次的分子设计迭代.

        Args:
            target: 目标属性, 如 {"HOMO-LUMO gap": "3.0 eV"}.
            n_rounds: 迭代轮数.
            n_per_round: 每轮生成的分子数.
            n_feedback: 每轮反馈的分子数.
            rag: RAGEngine 实例.
            run_name: 运行名称.
            runs_dir: 结果存储根目录.
            seed: 随机种子.
            max_tokens: LLM 最大 token 数.
            temperature: LLM 温度.
            gap_range_margin: gap 筛选范围边距 (eV).

        Returns:
            Dict with keys: run_dir, best, rounds, failed.
        """
        _ = evaluator

        target_gap = _parse_target_gap(target)
        store = ResultStore(
            run_dir=os.path.join(runs_dir, run_name),
            record_prompts=self.record_prompts,
        )
        all_round_feedbacks: List[Dict] = []
        global_best = None
        global_best_gap_diff = float("inf")

        t_run_start = time.monotonic()
        logger.info(f"=== SimpleLLMCore run: {run_name} ===")
        logger.info(f"  target={target}, rounds={n_rounds}, per_round={n_per_round}, seed={seed}")

        for round_id in range(1, n_rounds + 1):
            t_round_start = time.monotonic()
            logger.info(f"  Round {round_id}/{n_rounds}")

            # ── 1. RAG 检索 ──
            rag_results = None
            if self.use_rag and rag is not None and len(all_round_feedbacks) > 0:
                try:
                    rag_results = rag.query(
                        feedbacks=all_round_feedbacks,
                        n_samples=n_per_round,
                        gap_range_margin=gap_range_margin,
                    )
                except Exception as e:
                    logger.warning(f"  RAG query failed: {e}")

            # ── 2. Prompt 组装 ──
            messages = build_messages(
                task="molecular_design",
                target=target,
                rag_results=rag_results,
                round_feedbacks=all_round_feedbacks,
                n_per_round=n_per_round,
                reflection_mode=self.reflection_mode,
            )

            # ── 3. LLM 生成 ──
            t_llm_start = time.monotonic()
            raw_response, usage = self.provider.generate(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs,
            )
            t_llm = time.monotonic() - t_llm_start

            smiles_list = _parse_smiles(raw_response, n_per_round)
            logger.info(f"  Generated {len(smiles_list)} SMILES ({usage['total_tokens']} tokens, {t_llm:.1f}s)")

            # ── 5. SimpleEvaluator 评估 ──
            if smiles_list:
                try:
                    eval_df = self.evaluator.evaluate(
                        smiles_list=smiles_list,
                        target_gap=target_gap if target_gap is not None else 0.0,
                        n_top=n_feedback,
                    )
                except Exception as e:
                    logger.error(f"  SimpleEvaluator error: {e}")
                    eval_df = pd.DataFrame({"smiles": smiles_list, "xtb_success": [False] * len(smiles_list)})
            else:
                eval_df = pd.DataFrame({"smiles": [], "xtb_success": []})

            # ── 5b. 字段映射: xtb_* → dft_* ──
            eval_df = _map_xtb_to_dft(eval_df)
            if "dft_deviation_eV" not in eval_df.columns and target_gap is not None:
                if "dft_gap_eV" in eval_df.columns:
                    eval_df["dft_deviation_eV"] = (eval_df["dft_gap_eV"] - target_gap).abs()
            eval_df["target_gap_eV"] = target_gap

            # ── 6. 提取成功子集 ──
            dft_col = "dft_success"
            eval_success = eval_df[eval_df[dft_col] == True] if dft_col in eval_df.columns else pd.DataFrame()

            n_success = len(eval_success)
            feedback = _build_feedback(eval_df, round_id, n_feedback)

            # ── 7. 更新 RAG ──
            if self.use_rag and n_success > 0 and rag is not None:
                top_smiles = eval_success["smiles"].head(n_feedback).tolist()
                top_props = eval_success.head(n_feedback).to_dict("records")
                try:
                    rag.add_reflection(top_smiles, top_props)
                except Exception as e:
                    logger.warning(f"  RAG add_reflection failed: {e}")

            # ── 8. 记录最佳分子 ──
            if n_success > 0 and target_gap is not None:
                for _, row in eval_success.iterrows():
                    dft_gap = row.get("dft_gap_eV")
                    if dft_gap is None:
                        continue
                    diff = abs(dft_gap - target_gap)
                    if diff < global_best_gap_diff:
                        global_best_gap_diff = diff
                        global_best = row.to_dict()

            # ── 9. 落盘 ──
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
            logger.info(f"  Round {round_id} done: {n_success}/{len(smiles_list)} XTB ({t_round:.1f}s)")

        t_total = time.monotonic() - t_run_start
        config = {
            "provider": self.provider.model_name,
            "target": target,
            "n_rounds": n_rounds,
            "n_per_round": n_per_round,
            "n_feedback": n_feedback,
            "seed": seed,
            "max_tokens": max_tokens,
            "reflection_mode": self.reflection_mode,
            "run_name": run_name,
            "total_wall_time_s": round(t_total, 1),
            "simple_evaluator": True,
        }
        store.save_summary(config, {"best_molecule": global_best})

        logger.info(f"=== SimpleLLMCore run complete: {store.run_dir} ===")
        logger.info(f"  total: {t_total:.0f}s, best gap diff: {global_best_gap_diff:.4f}")

        return {
            "run_dir": store.run_dir,
            "best": global_best,
            "rounds": store._rounds,
            "failed": global_best is None,
        }
