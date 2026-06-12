"""分子光谱计算数据结构."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ExcitedState:
    index: int
    energy_eV: float
    wavelength_nm: float
    oscillator_strength: float
    symmetry: str = ""


@dataclass
class MoleculeSpectra:
    smiles: str
    xtb_gap_eV: Optional[float] = None
    xtb_homo_eV: Optional[float] = None
    xtb_lumo_eV: Optional[float] = None
    xtb_dipole_D: Optional[float] = None
    xtb_energy_Ha: Optional[float] = None
    xtb_success: bool = False

    dft_energy_Ha: Optional[float] = None
    dft_gap_eV: Optional[float] = None
    dft_homo_eV: Optional[float] = None
    dft_lumo_eV: Optional[float] = None
    dft_dipole_D: Optional[float] = None
    dft_mulliken_charges: Optional[List[float]] = None
    dft_success: bool = False

    excited_states: List[ExcitedState] = field(default_factory=list)
    nstates: int = 0
    td_dft_success: bool = False

    spectrum_wavelengths: Optional[np.ndarray] = None
    spectrum_intensities: Optional[np.ndarray] = None
    lambda_max_nm: Optional[float] = None

    # CIE 1931 感知色 (从透射率计算)
    cie_x: Optional[float] = None
    cie_y: Optional[float] = None

    error: Optional[str] = None

    @property
    def cie_deviation(self) -> Optional[float]:
        return getattr(self, "_cie_deviation", None)

    @cie_deviation.setter
    def cie_deviation(self, value):
        self._cie_deviation = value

    def get_excited_state_table(self):
        return [{"state": s.index, "energy_eV": round(s.energy_eV, 4),
                 "wavelength_nm": round(s.wavelength_nm, 2),
                 "oscillator_strength": round(s.oscillator_strength, 6)} for s in self.excited_states]

    def to_dict(self):
        d = {"smiles": self.smiles,
             "xtb_gap_eV": round(self.xtb_gap_eV, 4) if self.xtb_gap_eV is not None else None,
             "xtb_homo_eV": round(self.xtb_homo_eV, 4) if self.xtb_homo_eV is not None else None,
             "xtb_lumo_eV": round(self.xtb_lumo_eV, 4) if self.xtb_lumo_eV is not None else None,
             "xtb_dipole_D": round(self.xtb_dipole_D, 4) if self.xtb_dipole_D is not None else None,
             "xtb_energy_Ha": round(self.xtb_energy_Ha, 6) if self.xtb_energy_Ha is not None else None,
             "dft_gap_eV": round(self.dft_gap_eV, 4) if self.dft_gap_eV is not None else None,
             "dft_homo_eV": round(self.dft_homo_eV, 4) if self.dft_homo_eV is not None else None,
             "dft_lumo_eV": round(self.dft_lumo_eV, 4) if self.dft_lumo_eV is not None else None,
             "dft_dipole_D": round(self.dft_dipole_D, 4) if self.dft_dipole_D is not None else None,
             "dft_energy_Ha": round(self.dft_energy_Ha, 6) if self.dft_energy_Ha is not None else None,
             "dft_mulliken_charges": [round(c, 4) for c in self.dft_mulliken_charges] if self.dft_mulliken_charges else None,
             "cie_x": self.cie_x, "cie_y": self.cie_y,
             "lambda_max_nm": round(self.lambda_max_nm, 2) if self.lambda_max_nm is not None else None,
             "nstates": self.nstates, "excited_states": self.get_excited_state_table(),
             "xtb_success": self.xtb_success, "dft_success": self.dft_success, "td_dft_success": self.td_dft_success}
        return d
