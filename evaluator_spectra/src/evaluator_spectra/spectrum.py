"""吸收光谱 → 透射率 → CIE 1931 感知色.

物理流程:
    1. TD-DFT 激发态 (振子强度) → Gaussian 展宽 → 吸收光谱 A(λ)
    2. Beer-Lambert: 透射率 T(λ) = 10^(-c · Ã(λ))
      其中 Ã(λ) 归一化到 [0,1], c 为浓度×光程 (调节颜色饱和度)
    3. CIE 1931: 感知色 = CIE(T(λ))
"""

from typing import List, Optional, Tuple
import numpy as np
from .molecule import MoleculeSpectra


def gaussian_broaden(excited_states, fwhm=0.3, n_points=1000,
                     wavelength_range=(200, 800)):
    wl_min, wl_max = wavelength_range
    wavelengths = np.linspace(wl_min, wl_max, n_points)
    _NM_EV = 1239.84193
    energies_eV = _NM_EV / wavelengths
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    intensity = np.zeros_like(energies_eV)
    for state in excited_states:
        if state.oscillator_strength <= 0:
            continue
        pref = state.oscillator_strength / (sigma * np.sqrt(2 * np.pi))
        intensity += pref * np.exp(-0.5 * ((energies_eV - state.energy_eV) / sigma) ** 2)
    return wavelengths, intensity


def _absorbance_to_transmittance(wavelengths, absorbance, concentration=1.0):
    """Beer-Lambert: 吸收度 → 透射率.

    Args:
        absorbance: 原始吸收强度 (振子强度展宽, 任意单位).
        concentration: 浓度因子, 控制颜色饱和度.

    Returns:
        transmittance: [0, 1] 范围的透射率谱.
    """
    # 归一化吸收谱
    a_max = absorbance.max()
    if a_max == 0:
        return np.ones_like(absorbance)
    a_norm = absorbance / a_max
    # Beer-Lambert: T = 10^(-c · A_norm)
    transmittance = 10 ** (-concentration * a_norm)
    return transmittance


def compute_spectrum(mol, fwhm=0.3, n_points=1000, concentration=1.0):
    """计算吸收光谱 → 透射率 → CIE 1931 感知色."""
    from .cie1931 import spectrum_to_cie1931

    if not mol.td_dft_success or not mol.excited_states:
        return mol

    wl, abs_intensities = gaussian_broaden(
        mol.excited_states, fwhm=fwhm, n_points=n_points,
    )
    mol.spectrum_wavelengths = wl
    mol.spectrum_intensities = abs_intensities
    mol.lambda_max_nm = float(wl[abs_intensities.argmax()])

    # 透射率 → CIE 感知色
    trans = _absorbance_to_transmittance(wl, abs_intensities, concentration=concentration)
    cie = spectrum_to_cie1931(wl, trans)
    mol.cie_x = cie["cie_x"]
    mol.cie_y = cie["cie_y"]

    return mol
