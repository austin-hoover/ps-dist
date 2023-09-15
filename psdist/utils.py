import numpy as np
import scipy.special


def centers_from_edges(edges):
    """Compute bin centers from evenly spaced bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def edges_from_centers(centers):
    """Compute bin edges from evenly spaced bin centers."""
    delta = np.diff(centers)[0]
    return np.hstack([centers - 0.5 * delta, [centers[-1] + 0.5 * delta]])


def symmetrize(array):
    """Return a symmetrized version of array.

    array : A square upper or lower triangular matrix.
    """
    return array + array.T - np.diag(array.diagonal())


def random_selection(array, k):
    """Return k random elements of array without replacement.

    If 0 < k < 1, we select `k * len(array)` elements.
    """
    if type(array) in (list, tuple):
        array = np.array(array)
    if k is None:
        return array
    if k < 0 or k > array.shape[0]:
        raise ValueError("Number of samples must be < number of points.")
    if 0 < k < 1:
        k = k * array.shape[0]
    idx = np.random.choice(array.shape[0], int(k), replace=False)
    return array[idx]


def cov2corr(cov_mat):
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    return np.linalg.multi_dot([Dinv, cov_mat, Dinv])


def array_like(a):
    return np.ndim(np.array(a, dtype=object)) > 0


def sphere_surface_area(r=1.0, d=3):
    factor = 2.0 * np.pi ** (0.5 * (d + 1))
    factor = factor / scipy.special.gamma(0.5 * (d + 1))
    return factor * (r ** d)


def sphere_volume(r=1.0, d=3):
    factor = (np.pi ** (0.5 * d)) / scipy.special.gamma(1.0 + 0.5 * d)
    return factor * (r ** d)


def sphere_shell_volume(rmin=0.0, rmax=1.0, d=3):
    return sphere_volume(r=rmax, d=d) - sphere_volume(r=rmin, d=d)
