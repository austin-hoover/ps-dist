"""Accelerator physics functions."""
import numpy as np


# 1D (uncoupled) Courant-Snyder parameterization
# --------------------------------------------------------------------------------
def twiss_2x2(sigma):
    """RMS Twiss parameters from 2 x 2 covariance matrix.

    Parameters
    ----------
    sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    alpha : float
        The alpha parameter (-<uu'> / sqrt(<uu><u'u'> - <uu'>^2)).
    beta : float
        The beta parameter (<uu> / sqrt(<uu><u'u'> - <uu'>^2)).
    """
    eps = emittance_2x2(sigma)
    beta = sigma[0, 0] / eps
    alpha = -sigma[0, 1] / eps
    return alpha, beta


def emittance_2x2(sigma):
    """RMS emittance from u-u' covariance matrix.

    Parameters
    ----------
    sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    float
        The RMS emittance (sqrt(<uu><u'u'> - <uu'>^2)).
    """
    return np.sqrt(np.linalg.det(sigma))


def apparent_emittance(Sigma):
    """RMS apparent emittances from 2n x 2n covariance matrix.

    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    eps_x, eps_y, eps_z, ... : float
        The emittance in each phase-plane (eps_x, eps_y, eps_z, ...)
    """
    emittances = []
    for i in range(0, Sigma.shape[0], 2):
        emittances.append(emittance_2x2(Sigma[i : i + 2, i : i + 2]))
    if len(emittances) == 1:
        emittances = emittances[0]
    return emittances


def twiss(Sigma):
    """RMS Twiss parameters from 2n x 2n covariance matrix.

    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z, ... : float
        The Twiss parameters in each plane.
    """
    n = Sigma.shape[0] // 2
    params = []
    for i in range(n):
        j = i * 2
        params.extend(twiss_2x2(Sigma[j : j + 2, j : j + 2]))
    return params


def rotation_matrix(angle):
    """2 x 2 clockwise rotation matrix (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4x4(angle):
    """4 x 4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_adv_matrix(*phase_advances):
    """Phase advance matrix (clockwise rotation in each phase plane).

    Parameters
    ---------
    mu1, mu2, ..., mun : float
        The phase advance in each plane.

    Returns
    -------
    ndarray, shape (2n, 2n)
        Matrix which rotates x-x', y-y', z-z', etc. by the phase advances.
    """
    n = len(phase_advances)
    R = np.zeros((2 * n, 2 * n))
    for i, phase_advance in enumerate(phase_advances):
        i = i * 2
        R[i : i + 2, i : i + 2] = rotation_matrix(phase_advance)
    return R


def norm_matrix_2x2(alpha, beta):
    """2 x 2 normalization matrix for u-u'.

    Parameters
    ----------
    alpha : float
        The alpha parameter.
    beta : float
        The beta parameter.

    Returns
    -------
    V : ndarray, shape (2, 2)
        Matrix that transforms the ellipse defined by alpha/beta to a circle.
    """
    return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)


def norm_matrix(*twiss_params):
    """2n x 2n block-diagonal normalization matrix from Twiss parameters.

    Parameters
    ----------
    alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z, ... : float
        Twiss parameters for each dimension.

    Returns
    -------
    V : ndarray, shape (2n, 2n)
        Block-diagonal normalization matrix.
    """
    n = len(twiss_params) // 2
    V = np.zeros((2 * n, 2 * n))
    for i in range(n):
        j = i * 2
        V[j : j + 2, j : j + 2] = norm_matrix_2x2(*twiss_params[j : j + 2])
    return V


# Generalized (coupled) parameterizations
# --------------------------------------------------------------------------------


# Other
# --------------------------------------------------------------------------------
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
