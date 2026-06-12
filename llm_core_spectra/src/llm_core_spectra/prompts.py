"""Prompts for spectral targeting — high-saturation blue dye."""
from typing import Dict, List, Optional


def _describe_target(target_cie: tuple) -> str:
    x, y = target_cie
    return (
        "目标属性：\n"
        f"  目标感知色 CIE 1931: (x={x:.4f}, y={y:.4f}) — 高饱和蓝色\n"
        "  蓝光(450-490nm)被透射, 橙红光(580-620nm)被吸收\n"
        "\n"
        "分子设计指导：\n"
        "- 蓝色感知色需要分子在橙红区(580-620nm)有强吸收\n"
        "- 吸收峰应在 580-620nm, 对应 HOMO-LUMO gap≈2.0-2.15eV\n"
        "- 单吸收带即可, 不需要双吸收带\n"
        "- 振子强度越大越好, 确保橙红光被充分吸收\n"
        "- 中等大小的共轭体系即可达到该吸收范围\n"
        "- 给-吸电子(D-A)结构可有效调控吸收波长\n"
        "- HOMO-LUMO gap 应在 1.8-2.3 eV 范围内\n"
        "- 已知蓝色染料: 蒽醌类、酞菁类、三芳甲烷类"
    )


def _format_feedback_molecule(mol: dict, index: int) -> str:
    parts = [f"  {index}. {mol.get('smiles', '?')}"]
    for key, label in [("cie_x", "CIE"), ("lambda_max_nm", "λ_max"),
                        ("dft_gap_eV", "gap"), ("dft_dipole_D", "偶极")]:
        val = mol.get(key)
        if val is not None:
            parts.append(f"     {label}={val:.4f}" if isinstance(val, float) else f"     {label}={val}")
    states = mol.get("excited_states", [])
    if states:
        top = states[:5]
        ss = "; ".join(f"S{s['state']}:{s['wavelength_nm']:.0f}nm f={s['oscillator_strength']:.4f}" for s in top)
        parts.append(f"     激发态: {ss}")
    return "\n".join(parts)


def _format_round_feedback(feedbacks: List[Dict]) -> str:
    parts = ["以下是之前轮次的评估结果，请基于这些反馈改进设计："]
    for fb in feedbacks:
        parts.append(f"第 {fb.get('round','?')} 轮结果：")
        for i, mol in enumerate(fb.get("molecules", [])[:8], 1):
            parts.append(_format_feedback_molecule(mol, i))
        parts.append("")
    return "\n".join(parts)


def build_messages(target_cie, round_feedbacks=None, n_per_round=20, system_prompt=None):
    messages = []
    if system_prompt is None:
        system_prompt = (
            "你是一个分子光谱设计专家。根据目标属性，生成新的 SMILES 分子。\n"
            "要求：\n"
            "- 每行输出一个 SMILES，不要编号、不要解释、不要额外文字\n"
            "- 生成的分子应该结构合理、化学可行\n"
            "- 分子共轭体系越大，HOMO-LUMO gap 越小，吸收波长越长\n"
            "- 引入给/吸电子基团可以调控 gap\n"
            "- 如果之前轮次的结果给出了反馈，请基于反馈改进设计"
        )
    messages.append({"role": "system", "content": system_prompt})
    user = [_describe_target(target_cie)]
    if round_feedbacks:
        user.append(_format_round_feedback(round_feedbacks))
    user.append(f"请生成 {n_per_round} 个在橙红区有吸收的蓝色 SMILES 分子。")
    messages.append({"role": "user", "content": "\n".join(user)})
    return messages
