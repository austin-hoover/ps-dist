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


def normalize_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    ndim = eigenvectors.shape[0]
    U = unit_symplectic_matrix(ndim)
    for i in range(0, ndim, 2):
        v = eigenvectors[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigenvectors[:, i], eigenvectors[:, i + 1]) = (eigenvectors[:, i + 1], eigenvectors[:, i])
        eigenvectors[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigenvectors


def normalization_matrix_from_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    V = np.zeros(eigenvectors.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigenvectors[:, i].real
        V[:, i + 1] = (1.0j * eigenvectors[:, i]).real
    Vinv = np.linalg.inv(V)
    return Vinv
    

def normalization_matrix(cov: np.ndarray) -> np.ndarray:
    Sigma = cov
    U = unit_symplectic_matrix(ndim)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    Winv = normalization_matrix_from_eigenvectors(eigenvectors)
    return Winv


def emittance(cov: np.ndarray) -> float:
    """Compute emittance from covariance matrix."""
    return np.sqrt(np.linalg.det(cov))


def twiss_2x2(cov: np.ndarray) -> tuple[float]:
    """Compute twiss parameters from 2 x 2 covariance matrix.

    alpha = -<xx'> / sqrt(<xx><x'x'> - <xx'><xx'>)
    beta  =  <xx>  / sqrt(<xx><x'x'> - <xx'><xx'>)
    """
    eps = emittance(cov)
    beta = cov[0, 0] / eps
    alpha = -cov[0, 1] / eps
    return (alpha, beta)


def twiss(cov: np.ndarray) -> tuple[float]:
    """Compute two-dimensional twiss parameters from 2n x 2n covariance matrix.

    Parameters
    ----------
    cov : ndarray, shape (2n, 2n)
        Covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z, ... : float
        The Twiss parameters in each plane.
    """
    params = []
    for i in range(0, cov.shape[0], 2):
        params.extend(twiss_2x2(cov[i : i + 2, i : i + 2]))
    return params
    

def apparent_emittances(cov: np.ndarray) -> tuple[float]:
    """Compute rms apparent emittances from 2n x 2n covariance matrix.

    Parameters
    ----------
    cov : ndarray, shape (2n, 2n)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    eps_x, eps_y, eps_z, ... : float
        The emittance in each phase-plane (eps_x, eps_y, eps_z, ...)
    """
    emittances = []
    for i in range(0, cov.shape[0], 2):
        emittances.append(emittance_2x2(cov[i : i + 2, i : i + 2]))
    if len(emittances) == 1:
        emittances = emittances[0]
    return emittances
    

def intrinsic_emittances(cov: np.ndarray) -> tuple[float]:    
    """Compute rms intrinsic emittances from covariance matrix."""
    # To do: compute eigenvalues to extend to 6 x 6, rather than
    # using analytic eigenvalue solution specific to 4 x 4.
    Sigma = cov[:4, :4]
    U = unit_symplectic_matrix(ndim=4)
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(Sigma, U), 2))
    det_S = np.linalg.det(Sigma)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps_1, eps_2)    


def rotation_matrix_4x4_xy(angle: np.ndarray) -> np.ndarray:
    """4 x 4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_adv_matrix(*phase_advances) -> np.ndarray:
    """Phase advance matrix (clockwise rotation in each phase plane).

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


def norm_matrix_from_twiss_2x2(alpha: float, beta: float) -> np.ndarray:
    """2 x 2 normalization matrix for u-u'.

    Parameters
    ----------
    alpha : float
        The alpha parameter.
    beta : float
        The beta parameter.

    Returns
    -------
    ndarray, shape (2, 2)
        Matrix that transforms the ellipse defined by alpha/beta to a circle.
    """
    V = np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)
    return np.linalg.inv(V)


def norm_matrix_from_twiss(*twiss_params):
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
    ndim = len(twiss_params) // 2
    V = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        V[i : i + 2, i : i + 2] = norm_matrix_from_twiss_2x2(*twiss_params[i : i + 2])
    return np.linalg.inv(V)

