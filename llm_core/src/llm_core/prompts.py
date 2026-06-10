"""Prompt 组装 — 将目标属性 + RAG 上下文 + 历史反馈拼接为 LLM 可用 messages.

不预设任务目标，目标描述完全由调用方指定.
"""

from typing import Dict, List, Optional


def _format_number(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _describe_target(target: Dict[str, str]) -> str:
    lines = ["目标属性："]
    for key, val in target.items():
        lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def _format_rag_context(rag_results: List[Dict], max_lines: int = 50) -> str:
    if not rag_results:
        return "（无参考数据）"
    lines = [f"以下是从 QM9 数据库中检索到的 {min(len(rag_results), max_lines)} 个参考分子："]
    header = f"  {'SMILES':<35s} {'GAP(eV)':>8s} {'HOMO(eV)':>9s} {'LUMO(eV)':>9s} {'Dipole(D)':>9s}"
    lines.append(header)
    lines.append("  " + "-" * 75)
    for r in rag_results[:max_lines]:
        smiles = r.get("smiles", "?") or "?"
        if len(smiles) > 33:
            smiles = smiles[:30] + "..."
        lines.append(
            f"  {smiles:<35s} {_format_number(r.get('gap_eV')):>8s} "
            f"{_format_number(r.get('homo_eV')):>9s} "
            f"{_format_number(r.get('lumo_eV')):>9s} "
            f"{_format_number(r.get('dipole_D')):>9s}"
        )
    return "\n".join(lines)


def _format_round_feedback(
    round_feedbacks: List[Dict],
    target: Dict[str, str],
    reflection_mode: str = "spr",
) -> str:
    if not round_feedbacks:
        return ""

    scalar_keys = set()
    for k in target:
        kl = k.lower()
        if "homo-lumo" in kl or "gap" in kl:
            scalar_keys.add("dft_gap_eV")
        if "homo" in kl and "lumo" not in kl:
            scalar_keys.add("dft_homo_eV")
        if "lumo" in kl and "homo" not in kl:
            scalar_keys.add("dft_lumo_eV")
        if "dipole" in kl or "偶极" in k:
            scalar_keys.add("dft_dipole_D")
        if "energy" in kl or "能量" in k:
            scalar_keys.add("dft_energy_Ha")

    parts = []
    for fb in round_feedbacks:
        round_id = fb.get("round", "?")
        parts.append(f"第 {round_id} 轮结果：")
        molecules = fb.get("molecules", [])
        for i, mol in enumerate(molecules[:10], 1):
            smiles = mol.get("smiles", "?") or "?"
            success = "✓" if mol.get("dft_success") else "✗"

            if reflection_mode == "scalar" and scalar_keys:
                props = []
                for key in sorted(scalar_keys):
                    if key in mol and mol[key] is not None:
                        label = key.replace("dft_", "").replace("_eV", "")
                        props.append(f"{label}={mol[key]:.4f}")
                parts.append(f"  {i}. {smiles}  {', '.join(props)}  {success}")
            else:
                gap = mol.get("dft_gap_eV")
                homo = mol.get("dft_homo_eV")
                lumo = mol.get("dft_lumo_eV")
                dipole = mol.get("dft_dipole_D")
                energy = mol.get("dft_energy_Ha")
                props = []
                if gap is not None:
                    props.append(f"gap={gap:.4f}")
                if homo is not None:
                    props.append(f"HOMO={homo:.4f}")
                if lumo is not None:
                    props.append(f"LUMO={lumo:.4f}")
                if dipole is not None:
                    props.append(f"dipole={dipole:.4f}")
                if energy is not None:
                    props.append(f"E={energy:.6f}")
                parts.append(f"  {i}. {smiles}  {', '.join(props)}  {success}")

        if len(molecules) > 10:
            parts.append(f"  ... 共 {len(molecules)} 个分子")
        parts.append("")
    return "\n".join(parts)


def build_messages(
    task: str,
    target: Dict[str, str],
    rag_results: Optional[List[Dict]] = None,
    round_feedbacks: Optional[List[Dict]] = None,
    n_per_round: int = 20,
    system_prompt: Optional[str] = None,
    reflection_mode: str = "spr",
) -> List[Dict[str, str]]:
    """构建 LLM chat messages."""
    messages = []

    if system_prompt is None:
        system_prompt = (
            "你是一个分子设计专家。根据目标属性和提供的参考数据，生成新的 SMILES 分子。\n"
            "要求：\n"
            "- 每行输出一个 SMILES，不要编号、不要解释、不要额外文字\n"
            "- 生成的分子应该结构合理、化学可行\n"
            "- 如果之前轮次的结果给出了反馈，请基于反馈改进设计"
        )
    messages.append({"role": "system", "content": system_prompt})

    user_parts = [_describe_target(target)]

    if rag_results:
        user_parts.append("")
        user_parts.append(_format_rag_context(rag_results))

    if round_feedbacks:
        user_parts.append("")
        user_parts.append("以下是之前轮次的评估结果，请基于这些反馈改进设计：")
        user_parts.append("")
        user_parts.append(_format_round_feedback(round_feedbacks, target, reflection_mode))

    user_parts.append("")
    user_parts.append(f"请生成 {n_per_round} 个可能具有上述目标属性的新 SMILES 分子。")

    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages
