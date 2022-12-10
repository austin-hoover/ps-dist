"""Functions for bunches (point clouds)."""
import numpy as np

from . import ap
from . import utils


def cov(X):
    """Compute covariance matrix.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    
    Returns
    -------
    ndarray, shape (n, n)
        The covariance matrix of second-order moments.
    """
    return np.cov(X.T)


def apply(M, X):
    """Apply a linear transformation.
    
    Parameters
    ----------
    M : ndarray, shape (n, n)
        A matrix.
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
        
    Returns
    -------
    ndarray, shape (k, n)
        The transformed distribution.
    """
    return np.apply_along_axis(lambda v: np.matmul(M, v), 1, X)


def radial_extent(X, fraction=1.0):
    """Return radius of sphere containing fraction of points.
        
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    fraction : float
        Fraction of points in sphere.
        
    Returns
    -------
    radius : float
        Radius of sphere containing `fraction` of points.
    """
    k, n = X.shape
    radii = np.linalg.norm(X, axis=0)
    radii = np.sort(radii)
    if (k * fraction < 1.0):
        imax = k - 1
    else:
        imax = int(np.round(k * fraction))
    try:
        radius = radii[imax]
    except:
        radius = 0.0
    return radius


def slice_box(X, axis=None, center=None, width=None):
    """Return points within a box.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    center : ndarray, shape (n,)
        The center of the box.
    width : ndarray, shape (n,)
        The width of the box along each axis.
        
    Returns
    -------
    ndarray, shape (?, n)
        The points within the box.
    """    
    k, n = X.shape
    if type(axis) is int:
        axis = (axis,)
    if type(center) in [int, float]:
        center = np.full(n, center)
    if type(width) in [int, float]:
        width = np.full(n, width)
    center = np.array(center)
    width = np.array(width)            
    limits = list(zip(center - 0.5 * width, center + 0.5 * width))
    conditions = []
    for j, (umin, umax) in zip(axis, limits):
        conditions.append(X[:, j] > umin)
        conditions.append(X[:, j] < umax)
    idx = np.logical_and.reduce(conditions)
    return X[idx, :]


def slice_sphere(X, axis=0, r=None):
    """Return points within a sphere.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    r : float
        Radius of sphere.

    Returns
    -------
    ndarray, shape (?, n)
        The points within the sphere.
    """
    k, n = X.shape
    if axis is None:
        axis = tuple(range(n))
    if r is None:
        r = np.inf
    radii = np.linalg.norm(X[:, axis], axis=0)
    idx = radii < r
    return X[idx]


def slice_ellipsoid(X, axis=0, limits=None):
    """Return points within an ellipsoid.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    limits : list[float]
        Semi-axes of ellipsoid.

    Returns
    -------
    ndarray, shape (?, n)
        Points within the ellipsoid.
    """
    k, n = X.shape
    if axis is None:
        axis = tuple(range(n))
    if limits is None:
        limits = n * [np.inf]
    limits = np.array(limits)
    radii = np.sum((X[:, axis] / (0.5 * limits))**2, axis=1)
    idx = radii < 1.0
    return X[idx]


def histogram_bin_edges(X, bins=10, binrange=None):
    """Multi-dimensional histogram bin edges."""
    if type(bins) is not list:
        bins = X.shape[1] * [bins]
    if type(binrange) is not list:
        binrange = X.shape[1] * [binrange] 
    edges = [np.histogram_bin_edges(X[:, i], bins[i], binrange[i]) 
             for i in range(X.shape[1])]
    return edges
    
    
def histogram(X, bins=10, binrange=None, centers=False):
    """Multi-dimensional histogram."""
    edges = histogram_bin_edges(X, bins=bins, binrange=binrange)        
    hist, edges = np.histogramdd(X, bins=edges)
    if centers:
        return hist, [utils.get_centers(e) for e in edges]
    else:
        return hist, edges
    
    
def norm_xxp_yyp_zzp(X, scale_emittance=False):
    """Return coordinates normalized by x-x', y-y', z-z' Twiss parameters.
    
    Parameters
    ----------
    X : ndarray, shape (k, 6)
        Coordinates of k points in six-dimensional phase space.
    scale_emittance : bool
        Whether to divide the coordinates by the square root of the rms emittance.
    
    Returns
    -------
    Xn : ndarray, shape (N, 6)
        Normalized phase space coordinate array.
    """
    Sigma = np.cov(X.T)
    Xn = np.zeros(X.shape)
    for i in range(0, 6, 2):
        sigma = Sigma[i:i+2, i:i+2]
        alpha, beta = ap.twiss(sigma)
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        if scale_emittance:
            eps = ap.apparent_emittance(sigma)
            Xn[:, i:i+2] = Xn[:, i:i+2] / np.sqrt(eps)
    return Xn


def decorrelate(X):
    """Remove cross-plane correlations in the bunch by permuting 
    (x, x'), (y, y'), (z, z') pairs.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in six-dimensional phase space.
    
    Returns
    -------
    ndarray, shape (k, n)
        The decorrelated coordinate array.
    """
    if X.shape[1] != 6:
        raise ValueError('X must have 6 columns.')
    for i in (0, 2, 4):
        idx = np.random.permutation(np.arange(X.shape[0]))
        X[:, i:i+2] = X[idx, i:i+2]
    return X