"""分子对接数据模型."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DockingPose:
    """单个对接构象."""
    rank: int                       # 排名 (1=最佳)
    score: float                    # 对接分数 (kcal/mol, 越低越好)
    pdbqt_block: str = ""           # 对接后的配体 PDBQT


@dataclass
class Interaction:
    """一条蛋白-配体相互作用."""
    type: str                       # "hbond", "hydrophobic", "pi_stack", "salt_bridge", "halogen"
    residues: List[str] = field(default_factory=list)  # 参与的蛋白残基
    distance: Optional[float] = None
    angle: Optional[float] = None
    details: str = ""


@dataclass
class DockingResult:
    """一个分子的完整对接结果."""
    smiles: str
    best_score: Optional[float] = None       # 最佳对接分数
    all_scores: List[float] = field(default_factory=list)
    best_pose: Optional[DockingPose] = None
    interactions: List[Interaction] = field(default_factory=list)
    n_hbonds: int = 0
    n_hydrophobic: int = 0
    n_pi_stack: int = 0
    n_salt_bridges: int = 0
    docking_success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "smiles": self.smiles,
            "best_score": round(self.best_score, 4) if self.best_score is not None else None,
            "all_scores": [round(s, 4) for s in self.all_scores],
            "n_hbonds": self.n_hbonds,
            "n_hydrophobic": self.n_hydrophobic,
            "n_pi_stack": self.n_pi_stack,
            "n_salt_bridges": self.n_salt_bridges,
            "docking_success": self.docking_success,
        }
        if self.interactions:
            d["interactions"] = str("; ".join(
                f"{it.type}({','.join(it.residues)})" for it in self.interactions
            ))
        else:
            d["interactions"] = ""
        if self.best_pose and self.best_pose.pdbqt_block:
            d["best_pose_pdbqt"] = self.best_pose.pdbqt_block
        return d
