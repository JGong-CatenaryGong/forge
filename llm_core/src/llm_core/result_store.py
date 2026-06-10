"""ResultStore — 每轮实验结果实时落盘.

目录布局:
    runs/{name}/
    ├── feedback_table.csv     # 累积反馈表格 (每轮更新)
    ├── manifest.json          # 总览
    ├── round_01/
    │   ├── prompt.json
    │   ├── llm_response.txt
    │   ├── usage.json
    │   ├── eval_results.parquet
    │   └── rag_context.json
    ├── round_02/ ...
    └── summary.json
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DISPLAY_COLS = ["round", "rank", "smiles", "dft_gap_eV", "deviation_eV",
                "dft_homo_eV", "dft_lumo_eV", "dft_dipole_D", "dft_energy_Ha"]


class ResultStore:
    """实验结果持久化管理器."""

    def __init__(self, run_dir: str, record_prompts: bool = True):
        self.run_dir = run_dir
        self.record_prompts = record_prompts
        self._rounds: List[Dict] = []
        self._cumulative_feedback: List[Dict] = []
        os.makedirs(run_dir, exist_ok=True)

    # ---- 每轮保存 ----

    def save_round(
        self,
        round_id: int,
        prompt_messages: List[Dict],
        llm_response: str,
        llm_usage: Dict,
        eval_df: pd.DataFrame,
        rag_context: Optional[List[Dict]] = None,
        smiles_generated: Optional[List[str]] = None,
    ) -> None:
        """保存一轮实验结果到磁盘，并更新累积反馈表."""
        rd = os.path.join(self.run_dir, f"round_{round_id:02d}")
        os.makedirs(rd, exist_ok=True)

        if self.record_prompts and prompt_messages:
            with open(os.path.join(rd, "prompt.json"), "w") as f:
                json.dump(prompt_messages, f, ensure_ascii=False, indent=2)

        with open(os.path.join(rd, "llm_response.txt"), "w") as f:
            f.write(llm_response)

        with open(os.path.join(rd, "usage.json"), "w") as f:
            json.dump(llm_usage, f, indent=2)

        eval_df.to_parquet(os.path.join(rd, "eval_results.parquet"), index=False)

        if rag_context:
            with open(os.path.join(rd, "rag_context.json"), "w") as f:
                json.dump(rag_context, f, ensure_ascii=False, indent=2)

        if smiles_generated:
            with open(os.path.join(rd, "generated_smiles.txt"), "w") as f:
                f.write("\n".join(smiles_generated))

        # 内存汇总
        n_success = int(eval_df["dft_success"].sum()) if "dft_success" in eval_df.columns else 0
        n_failed = int((~eval_df["dft_success"]).sum()) if "dft_success" in eval_df.columns else len(eval_df)
        self._rounds.append({
            "round": round_id,
            "n_generated": len(smiles_generated) if smiles_generated else 0,
            "n_eval_success": n_success,
            "n_eval_failed": n_failed,
            "prompt_tokens": llm_usage.get("prompt_tokens", 0),
            "completion_tokens": llm_usage.get("completion_tokens", 0),
            "time_s": llm_usage.get("time_s", 0),
        })

        # 更新累积反馈表 (仅 DFT 成功分子)
        if "dft_success" in eval_df.columns and n_success > 0:
            ok = eval_df[eval_df["dft_success"] == True].copy()
            ok["round"] = round_id
            ok["rank"] = range(1, len(ok) + 1)
            for _, row in ok.iterrows():
                self._cumulative_feedback.append(row.to_dict())

        # 写累积 CSV (不做列筛选，完整保存，便于后续 filter)
        if self._cumulative_feedback:
            fb_df = pd.DataFrame(self._cumulative_feedback)
            fb_path = os.path.join(self.run_dir, "feedback_table.csv")
            # 选展示列
            cols = [c for c in DISPLAY_COLS if c in fb_df.columns]
            fb_df[cols].to_csv(fb_path, index=False, float_format="%.4f")

        logger.info(f"Round {round_id} saved: {n_success}/{len(eval_df)} DFT ok")

    # ---- 实验汇总 ----

    def save_summary(self, config: Dict, final_stats: Optional[Dict] = None) -> str:
        """保存实验汇总 manifest + summary."""
        total_prompt_tokens = sum(r["prompt_tokens"] for r in self._rounds)
        total_completion_tokens = sum(r["completion_tokens"] for r in self._rounds)
        total_tokens = total_prompt_tokens + total_completion_tokens
        total_llm_time = sum(r["time_s"] for r in self._rounds)
        total_wall_time = config.get("total_wall_time_s", 0)
        total_generated = sum(r["n_generated"] for r in self._rounds)
        total_eval_success = sum(r["n_eval_success"] for r in self._rounds)
        total_eval_failed = sum(r["n_eval_failed"] for r in self._rounds)

        manifest = {
            "config": config,
            "total_rounds": len(self._rounds),
            "rounds": self._rounds,
            "totals": {
                "total_wall_time_s": round(total_wall_time, 1),
                "total_llm_time_s": round(total_llm_time, 1),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_generated": total_generated,
                "total_dft_success": total_eval_success,
                "total_dft_failed": total_eval_failed,
                "dft_failure_rate": round(total_eval_failed / max(total_generated, 1), 4),
                "any_success": total_eval_success > 0,
            },
            "run_failed": total_eval_success == 0,
        }

        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        summary = {
            "config_summary": {k: str(v) for k, v in config.items()},
            **(final_stats or {}),
            "totals": manifest["totals"],
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 人可读的时间/token 摘要
        timing_path = os.path.join(self.run_dir, "timing.txt")
        with open(timing_path, "w") as f:
            f.write(f"总耗时:           {total_wall_time:.0f}s ({total_wall_time/60:.1f}min)\n")
            f.write(f"LLM 总耗时:       {total_llm_time:.0f}s ({total_llm_time/60:.1f}min)\n")
            f.write(f"LLM 总 tokens:    {total_tokens} (prompt={total_prompt_tokens}, completion={total_completion_tokens})\n")
            f.write(f"总轮次:           {len(self._rounds)}\n")
            f.write(f"总生成分子:       {total_generated}\n")
            f.write(f"DFT 成功/失败:    {total_eval_success}/{total_eval_failed}\n")

        logger.info(f"Summary saved: {self.run_dir}")
        if total_wall_time > 0:
            logger.info(f"  wall={total_wall_time:.0f}s, llm={total_llm_time:.0f}s, tokens={total_tokens}")

        return self.run_dir

    # ---- 反馈表格 ----

    def feedback_table(self, target_gap: Optional[float] = None) -> pd.DataFrame:
        """汇总所有轮次的反馈分子为一张表格."""
        if not self._cumulative_feedback:
            return pd.DataFrame()
        fb_df = pd.DataFrame(self._cumulative_feedback)
        if target_gap is not None and "dft_gap_eV" in fb_df.columns:
            fb_df["deviation_eV"] = (fb_df["dft_gap_eV"] - target_gap).abs()
        cols = [c for c in DISPLAY_COLS if c in fb_df.columns]
        return fb_df[cols].sort_values(["round", "rank"]).reset_index(drop=True)

    @staticmethod
    def read_feedback_table(run_dir: str, target_gap: Optional[float] = None) -> pd.DataFrame:
        """从已保存的 run 目录读取反馈表格."""
        csv_path = os.path.join(run_dir, "feedback_table.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if target_gap is not None and "dft_gap_eV" in df.columns:
                df["deviation_eV"] = (df["dft_gap_eV"] - target_gap).abs()
            return df
        # 回退: 扫描 round 目录
        import glob
        parquet_files = sorted(glob.glob(os.path.join(run_dir, "round_*/eval_results.parquet")))
        if not parquet_files:
            return pd.DataFrame()
        all_rows = []
        for pf in parquet_files:
            rd = int(os.path.basename(os.path.dirname(pf)).split("_")[1])
            df = pd.read_parquet(pf)
            if "dft_success" in df.columns:
                df = df[df["dft_success"] == True]
            if len(df) == 0:
                continue
            df = df.copy()
            df["round"] = rd
            df["rank"] = range(1, len(df) + 1)
            all_rows.append(df)
        combined = pd.concat(all_rows, ignore_index=True)
        if target_gap is not None and "dft_gap_eV" in combined.columns:
            combined["deviation_eV"] = (combined["dft_gap_eV"] - target_gap).abs()
        cols = [c for c in DISPLAY_COLS if c in combined.columns]
        return combined[cols].sort_values(["round", "rank"]).reset_index(drop=True)
