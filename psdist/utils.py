import numpy as np


def get_centers(edges):
    """Compute bin centers from bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def get_edges(centers):
    """Compute bin edges from bin centers."""
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
    array_copy = np.copy(array)
    if 0 < k < array_copy.shape[0]:
        if k < 1:
            k = k * array_copy.shape[0]
        idx = np.random.choice(array_copy.shape[0], int(k), replace=False)
        array_copy = array_copy[idx]
    return array_copy


def cov2corr(cov_mat):
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    return np.linalg.multi_dot([Dinv, cov_mat, Dinv])