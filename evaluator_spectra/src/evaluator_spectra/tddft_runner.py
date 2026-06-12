"""ORCA TD-DFT (PBE0/def2-SVP, RIJCOSX) runner."""
import logging, os, re, subprocess, tempfile
from typing import List, Optional
from .molecule import ExcitedState, MoleculeSpectra

logger = logging.getLogger(__name__)
_NM_EV = 1239.84193


def _make_orca_input(xyz_block, nstates=10, functional="PBE0", basis="def2-SVP"):
    return f"""! {functional} {basis} RIJCOSX def2/J
%pal
  nprocs 8
end
%maxcore 4000
%tddft
  nroots {nstates}
  tda false
end
* xyz 0 1
{xyz_block}
*
"""


def _parse_orca_output(text):
    result = {}
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    if m: result["energy_Ha"] = float(m.group(1))

    orbs = re.findall(r"^\s*(\d+)\s+([\d.]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", text, re.MULTILINE)
    occ = [(int(i), float(e)) for i, o, _, e in orbs if float(o) > 0.5]
    virt = [(int(i), float(e)) for i, o, _, e in orbs if float(o) < 0.1]
    if occ: result["homo_eV"] = occ[-1][1]
    if virt: result["lumo_eV"] = virt[0][1]
    if "homo_eV" in result and "lumo_eV" in result:
        result["gap_eV"] = result["lumo_eV"] - result["homo_eV"]

    m = re.search(r"Dipole Moment.*?x\s*:\s*(-?\d+\.\d+).*?y\s*:\s*(-?\d+\.\d+).*?z\s*:\s*(-?\d+\.\d+)", text, re.DOTALL)
    if m:
        d = (float(m.group(1))**2 + float(m.group(2))**2 + float(m.group(3))**2)**0.5
        result["dipole_D"] = d

    m = re.search(r"MULLIKEN ATOMIC CHARGES\s+(.*?)(?:\n\n|\n\s*\n)", text, re.DOTALL)
    if m:
        charges = []
        for line in m.group(1).split("\n"):
            p = line.strip().split()
            if len(p) >= 3 and ":" in p:
                try: charges.append(float(p[-1]))
                except: pass
        if charges: result["mulliken_charges"] = charges

    # Excited states
    state_energies = {}
    for m in re.finditer(r"STATE\s+(\d+):\s+E\s*=\s*([\d.]+)\s+au\s+([\d.]+)\s+eV", text):
        state_energies[int(m.group(1))] = float(m.group(3))

    excited = []
    if state_energies:
        for line in text.split("\n"):
            if "->" not in line: continue
            parts = line.strip().split()
            if len(parts) < 7: continue
            try:
                n = int(parts[2].split("-")[0])
                if n in state_energies:
                    osc = float(parts[6])
                    wl = _NM_EV / state_energies[n] if state_energies[n] > 0 else 0
                    excited.append(ExcitedState(index=n, energy_eV=state_energies[n],
                                                wavelength_nm=wl, oscillator_strength=osc))
            except (ValueError, IndexError): continue

    if excited:
        result["excited_states"] = excited
        result["nstates"] = len(excited)
    return result


def _run_xtb_single_to_xyz(smiles, xtb_bin, gfn=2, omp_threads=4, timeout=300):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    r = AllChem.EmbedMolecule(mol, randomSeed=42)
    if r != 0:
        r = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if r != 0: return None
    try: AllChem.MMFFOptimizeMolecule(mol)
    except:
        try: AllChem.UFFOptimizeMolecule(mol)
        except: pass
    conf = mol.GetConformer()
    xyz = [str(mol.GetNumAtoms()), ""]
    for i, a in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        xyz.append(f"{a.GetSymbol():2s}  {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    xb = "\n".join(xyz)
    with tempfile.TemporaryDirectory(prefix="xtb_") as tmp:
        with open(os.path.join(tmp, "input.xyz"), "w") as f: f.write(xb)
        env = {**os.environ, "OMP_NUM_THREADS": str(omp_threads)}
        proc = subprocess.run([xtb_bin, os.path.join(tmp, "input.xyz"), "--gfn", str(gfn), "--opt", "--dipole"],
                              capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env)
        if proc.returncode != 0: return None
        opt = os.path.join(tmp, "xtbopt.xyz")
        if not os.path.exists(opt): return None
        with open(opt) as f:
            lines = f.readlines()
        if len(lines) >= 2: lines[1] = "energy\n"
        return "".join(lines)


def _parse_xtb_stdout(stdout):
    result = {}
    m = re.search(r"\|\s*HOMO-LUMO\s+GAP\s+([\d.]+)\s+eV\s*\|", stdout)
    if m: result["gap_eV"] = float(m.group(1))
    m = re.search(r"\|\s*TOTAL\s+ENERGY\s+([\-\d.Ee]+)\s+Eh\s*\|", stdout)
    if m: result["total_energy_Eh"] = float(m.group(1))
    lines = stdout.split("\n")
    hi, li = -1, -1
    for i, l in enumerate(lines):
        if "(HOMO)" in l: hi = i
        if "(LUMO)" in l: li = i
    try:
        if hi >= 0: result["homo_eV"] = float(lines[hi].split()[lines[hi].split().index("(HOMO)") - 1])
        if li >= 0: result["lumo_eV"] = float(lines[li].split()[lines[li].split().index("(LUMO)") - 1])
    except: pass
    for line in lines:
        if line.strip().startswith("full:"):
            p = line.split()
            if len(p) >= 5:
                try: result["dipole_D"] = float(p[4])
                except: pass
    return result


class TDDTFTRunner:
    def __init__(self, xtb_bin=None, xtb_gfn=2, xtb_omp_threads=4, xtb_timeout=300,
                 orca_bin="/home/jgong/calculations/orca/orca",
                 functional="PBE0", basis="def2-SVP", nstates=10, orca_timeout=600):
        self.xtb_bin = xtb_bin or os.environ.get("XTB_BIN", "/home/jgong/calculations/xtb-dist/bin/xtb")
        self.xtb_gfn, self.xtb_omp_threads, self.xtb_timeout = xtb_gfn, xtb_omp_threads, xtb_timeout
        self.orca_bin = orca_bin
        self.functional, self.basis = functional, basis
        self.nstates, self.orca_timeout = nstates, orca_timeout

    def run(self, smiles):
        mol = MoleculeSpectra(smiles=smiles)
        try:
            opt_xyz = _run_xtb_single_to_xyz(smiles, self.xtb_bin, self.xtb_gfn,
                                              self.xtb_omp_threads, self.xtb_timeout)
            if opt_xyz is None:
                mol.error = "XTB failed"; return mol

            try:
                with tempfile.TemporaryDirectory(prefix="xtbp_") as tmp:
                    with open(os.path.join(tmp, "_i.xyz"), "w") as f: f.write(opt_xyz)
                    env = {**os.environ, "OMP_NUM_THREADS": str(self.xtb_omp_threads)}
                    p = subprocess.run([self.xtb_bin, os.path.join(tmp, "_i.xyz"), "--gfn", str(self.xtb_gfn), "--sp", "--dipole"],
                                       capture_output=True, text=True, timeout=self.xtb_timeout, cwd=tmp, env=env)
                    if p.returncode == 0:
                        d = _parse_xtb_stdout(p.stdout)
                        mol.xtb_gap_eV = d.get("gap_eV"); mol.xtb_energy_Ha = d.get("total_energy_Eh")
                        mol.xtb_homo_eV = d.get("homo_eV"); mol.xtb_lumo_eV = d.get("lumo_eV")
                        mol.xtb_dipole_D = d.get("dipole_D"); mol.xtb_success = True
            except Exception as e: logger.warning(f"XTB parse: {e}")

            with tempfile.TemporaryDirectory(prefix="orca_") as tmp:
                xyz_lines = opt_xyz.strip().split("\n")
                inp = _make_orca_input("\n".join(xyz_lines[2:]), nstates=self.nstates,
                                       functional=self.functional, basis=self.basis)
                with open(os.path.join(tmp, "input.inp"), "w") as f: f.write(inp)
                proc = subprocess.run([self.orca_bin, os.path.join(tmp, "input.inp")],
                                      cwd=tmp, timeout=self.orca_timeout, capture_output=True, text=True)
                if proc.returncode != 0:
                    mol.error = f"ORCA exit {proc.returncode}: {proc.stderr[-400:]}"
                    return mol
                parsed = _parse_orca_output(proc.stdout)
                mol.dft_energy_Ha = parsed.get("energy_Ha")
                mol.dft_homo_eV = parsed.get("homo_eV"); mol.dft_lumo_eV = parsed.get("lumo_eV")
                mol.dft_gap_eV = parsed.get("gap_eV"); mol.dft_dipole_D = parsed.get("dipole_D")
                mol.dft_mulliken_charges = parsed.get("mulliken_charges"); mol.dft_success = True
                exc = parsed.get("excited_states", [])
                if exc:
                    mol.excited_states = exc; mol.nstates = len(exc); mol.td_dft_success = True
                else:
                    mol.error = "No excited states found"

        except subprocess.TimeoutExpired: mol.error = f"ORCA timeout ({self.orca_timeout}s)"
        except FileNotFoundError: mol.error = f"ORCA not found: {self.orca_bin}"
        except Exception as e: mol.error = f"{type(e).__name__}: {e}"; logger.error(f"TDDTFTRunner error ({smiles}): {e}")
        return mol
