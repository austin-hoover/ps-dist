import numpy as np


def get_centers(edges):
    """Compute bin centers from bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def get_edges(centers):
    """Compute bin edges from bin centers."""
    delta = np.diff(centers)[0]
    return np.hstack([centers - 0.5 * delta, [centers[-1] + 0.5 * delta]])


def symmetrize(a):
    """Return a symmetrized version of matrix.
    
    a : A square upper or lower triangular matrix.
    """
    return a + a.T - np.diag(a.diagonal())
    
    
def rand_rows(X, k):
    """Return k random elements of X."""
    Xsamp = np.copy(X)
    if k < len(X):
        idx = np.random.choice(Xsamp.shape[0], k, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp


def cov2corr(cov_mat):
    """Compute correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    return np.linalg.multi_dot([Dinv, cov_mat, Dinv])