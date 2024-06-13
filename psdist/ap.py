"""Accelerator physics module."""

import numpy as np

from .utils import rotation_matrix


def lorentz_factors(mass: float, kin_energy: float) -> float:
    """Compute relativistic factors from mass and energy.

    Parameters
    ----------
    mass : float
        Particle mass divided by c^2 (units of energy).
    kin_energy : float
        Particle kinetic energy.

    Returns
    -------
    gamma, beta : float
        beta = absolute velocity divided by the speed of light
        gamma = sqrt(1 - (1/beta)**2)
    """
    gamma = 1.0 + (kin_energy / mass)
    beta = np.sqrt(1.0 - (1.0 / (gamma**2)))
    return gamma, beta


def unit_symplectic_matrix(ndim: int = 4) -> np.ndarray:
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    ndim = eigvecs.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigvecs[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigvecs[:, i], eigvecs[:, i + 1]) = (eigvecs[:, i + 1], eigvecs[:, i])
        eigvecs[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigvecs


def normalization_matrix_from_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1.0j * eigvecs[:, i]).real
    return np.linalg.inv(V)


def rotation_matrix_4x4_xy(angle: float) -> np.ndarray:
    """4 x 4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane (angle in radians)."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_adv_matrix(*phase_advances) -> np.ndarray:
    """Phase advance matrix (clockwise rotation in each two-dimensional phase plane).

    Parameters
    ---------
    mu1, mu2, ..., mun : float
        The phase advance in each plane.

    Returns
    -------
    ndarray, shape (2n, 2n)
        Rotates x-x', y-y', z-z', etc. by the phase advances.
    """
    ndim = 2 * len(phase_advances)
    R = np.zeros((ndim, ndim))
    for i, phase_advance in enumerate(phase_advances):
        j = i * 2
        R[j : j + 2, j : j + 2] = rotation_matrix(phase_advance)
    return R
