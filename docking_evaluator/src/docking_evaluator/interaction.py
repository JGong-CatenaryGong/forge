"""蛋白-配体相互作用分析 — ProLIF 2.x (MDAnalysis + RDKit)."""

import logging, os, tempfile
import MDAnalysis as mda
import prolif
from .molecule import DockingResult, Interaction

logger = logging.getLogger(__name__)

# AutoDock 原子类型 → 元素符号
_AD_TO_ELEMENT = {
    "C": "C", "A": "C",
    "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S",
    "P": "P",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "H": "H", "HD": "H",
    "Fe": "Fe", "Zn": "Zn", "Mg": "Mg", "Ca": "Ca",
}


def _fix_element(line: str) -> str:
    """给 PDB 行补充正确的元素列 (cols 77-78)."""
    if len(line) < 78:
        return line
    parts = line.split()
    ad_type = parts[-1] if parts else ""
    elem = _AD_TO_ELEMENT.get(ad_type, line[12:14].strip() or "C")
    return line[:76] + f"{elem:>2}" + line[78:] if len(line) > 78 else line[:76] + f"{elem:>2}"


def analyze_interactions(result: DockingResult, receptor_pdb: str, method=None) -> DockingResult:
    if not result.docking_success or result.best_pose is None:
        return result

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        with open(receptor_pdb) as rf:
            for line in rf:
                if line.startswith(("ATOM", "HETATM", "TER")):
                    f.write(_fix_element(line))
        f.write("TER\n")
        for line in result.best_pose.pdbqt_block.split("\n"):
            if line.startswith(("ATOM", "HETATM")):
                f.write(_fix_element(line))
        f.write("END\n")
        cpath = f.name

    try:
        u = mda.Universe(cpath)
        prot = u.select_atoms("protein")
        lig = u.select_atoms("not protein and not resname HOH and not resname NDP")
        if lig.n_atoms == 0:
            lig = u.select_atoms("not protein")

        m_prot = prolif.Molecule.from_mda(prot)
        m_lig = prolif.Molecule.from_mda(lig)
        fp = prolif.Fingerprint()
        fp.run_from_iterable([m_lig], m_prot)
        df = fp.to_dataframe()

        for col in df.columns:
            if not df[col].any():
                continue
            itype = col[2].lower()
            residues = [f"{col[1]}-{col[0]}"]
            result.interactions.append(Interaction(type=itype, residues=residues))

        for it in result.interactions:
            t = it.type.lower()
            if any(x in t for x in ("hb", "hbond")):
                result.n_hbonds += 1
            elif any(x in t for x in ("vdw", "hydrophobic")):
                result.n_hydrophobic += 1
            elif any(x in t for x in ("pi", "pistack", "pication")):
                result.n_pi_stack += 1
            elif any(x in t for x in ("salt", "ionic")):
                result.n_salt_bridges += 1
    except Exception as e:
        logger.error(f"ProLIF error: {e}")
    finally:
        os.unlink(cpath)

    return result
