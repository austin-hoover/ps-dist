"""Accelerator physics functions."""
import numpy as np
from numpy import linalg as la


def twiss_2x2(sigma):
    """Return rms Twiss parameters from u-u' covariance matrix."""
    eps = emittance_2x2(sigma)
    beta = sigma[0, 0] / eps
    alpha = -sigma[0, 1] / eps
    return alpha, beta


def emittance_2x2(sigma):
    """Return rms emittance from u-u' covariance matrix."""
    return np.sqrt(la.det(sigma))


def apparent_emittance(Sigma):
    """Return eps_x, eps_y, eps_z."""
    emittances = []
    for i in range(0, Sigma.shape[0], 2):
        emittances.append(emittance_2x2(Sigma[i : i + 2, i : i + 2]))
    if len(emittances) == 1:
        emittances = emittances[0]
    return emittances


def twiss(Sigma):
    """Return alpha_x, beta_x, alpha_y, beta_y, ..."""
    n = Sigma.shape[0] // 2
    params = []
    for i in range(n):
        j = i * 2
        params.extend(twiss_2x2(Sigma[j : j + 2, j : j + 2]))
    return params


def rotation_mat(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_mat_4x4(angle):
    """4x4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_adv_mat(*phase_advances):
    n = len(phase_advances)
    R = np.zeros((2 * n, 2 * n))
    for i, phase_advance in enumerate(phase_advances):
        i = i * 2
        R[i : i + 2, i : i + 2] = rotation_mat(phase_advance)
    return R


def norm_mat_2x2(alpha, beta):
    """Normalization matrix for u-u'."""
    return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)


def norm_mat(*twiss_params):
    """Order is (alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z).
    Leave out the dimensions you don't want."""
    n = len(twiss_params) // 2
    V = np.zeros((2 * n, 2 * n))
    for i in range(n):
        j = i * 2
        V[j : j + 2, j : j + 2] = norm_mat_2x2(*twiss_params[j : j + 2])
    return V


def lorentz_factors(mass=1.0, kin_energy=1.0):
    """Return relativistic factors gamma and beta.

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
