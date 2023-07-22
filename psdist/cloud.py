"""Functions for point clouds."""
import collections

import numpy as np
import scipy.interpolate
import scipy.special
import scipy.stats

from psdist import ap
import psdist.utils as utils
from psdist.utils import array_like
from psdist.utils import centers_from_edges
from psdist.utils import cov2corr
from psdist.utils import random_selection


# Analysis
# ------------------------------------------------------------------------------
def mean(X):
    """Compute mean (centroid).

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.

    Returns
    -------
    ndarray, shape (d,)
        The centroid coordinates.
    """
    return np.mean(X, axis=0)


def cov(X):
    """Compute covariance matrix.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.

    Returns
    -------
    ndarray, shape (d, d)
        The covariance matrix of second-order moments.
    """
    return np.cov(X.T)


def corr(X):
    """Compute correlation matrix.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.

    Returns
    -------
    ndarray, shape (d, d)
        The correlation matrix.
    """
    return cov2corr(np.cov(X.T))


def get_radii(X):
    return np.linalg.norm(X, axis=1)


def get_ellipsoid_radii(X):
    Sigma_inv = np.linalg.inv(np.cov(X.T))
    func = lambda point: np.sqrt(np.linalg.multi_dot([point.T, Sigma_inv, point]))
    return transform(X, func)


def enclosing_sphere(X, axis=None, fraction=1.0):
    """Scales sphere until it contains some fraction of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The distribution is projected onto this axis before proceeding. The
        ellipsoid is defined in this subspace.
    fraction : float
        Fraction of points in sphere.

    Returns
    -------
    radius : float
        The sphere radius.
    """
    radii = np.sort(get_radii(project(X, axis)))
    index = int(np.round(X.shape[0] * fraction)) - 1
    return radii[index]


def enclosing_ellipsoid(X, axis=None, fraction=1.0):
    """Scale the rms ellipsoid until it contains some fraction of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The distribution is projected onto this axis before proceeding. The
        ellipsoid is defined in this subspace.
    fraction : float
        Fraction of points enclosed.

    Returns
    -------
    float
        The ellipsoid "radius" (x^T Sigma^-1 x) relative to the rms ellipsoid.
    """
    radii = np.sort(get_ellipsoid_radii(project(X, axis)))
    index = int(np.round(X.shape[0] * fraction)) - 1
    return radii[index]


## Statistical distance measures (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.106.065302)
## [...]


# Transformation
# ------------------------------------------------------------------------------


def project(X, axis=None):
    """Axis-aligned projection. (Just calls `X[:, axis]`.)

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple[int], length l
        The axis on which to project the points.

    Returns
    -------
    ndarray, shape (n, l)
        The points projected onto the specified axis.
    """
    if axis is None:
        axis = tuple(np.arange(X.shape[1]))
    if array_like(axis) and len(axis) > X.shape[1]:
        raise ValueError("Invalid projection axis.")
    return X[:, axis]


def transform(X, func=None, **kws):
    """Apply a nonlinear transformation.

    This function just calls `np.apply_along_axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    function : callable
        Function applied to each point in X. Call signature is
        `function(point, **kws)` where `point` is an n-dimensional
        point given by one row of `X`.
    **kws
        Key word arguments for

    Returns
    -------
    ndarray, shape (n, d)
        The transformed distribution.
    """
    return np.apply_along_axis(lambda point: func(point, **kws), 1, X)


def transform_linear(X, M):
    """Apply a linear transformation.

    This function just calls `np.apply_along_axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    M : ndarray, shape (d, d)
        A linear transfer matrix.

    Returns
    -------
    ndarray, shape (n, d)
        The transformed distribution.
    """
    func = lambda point: np.matmul(M, point)
    return transform(X, lambda point: np.matmul(M, point))


def shift(X, delta=0.0):
    return X + delta


def scale(X, factor=1.0):
    return X * factor


def slice_planar(X, axis=None, center=None, width=None, limits=None):
    """Return points within a planar slice.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    center : ndarray, shape (n,)
        The center of the box.
    width : ndarray, shape (d,)
        The width of the box along each axis.
    limits : ndarray, shape (d, 2)
        The (min, max) along each axis. Overrides `center` and `edges` if provided.

    Returns
    -------
    ndarray, shape (?, n)
        The points within the box.
    """
    n, d = X.shape
    if not array_like(axis):
        axis = (axis,)
    if limits is None:
        if not array_like(center):
            center = np.full(d, center)
        if not array_like(width):
            width = np.full(d, width)
        center = np.array(center)
        width = np.array(width)
        limits = list(zip(center - 0.5 * width, center + 0.5 * width))
    limits = np.array(limits)
    if limits.ndim == 1:
        limits = limits[None, :]
    conditions = []
    for j, (umin, umax) in zip(axis, limits):
        conditions.append(X[:, j] > umin)
        conditions.append(X[:, j] < umax)
    idx = np.logical_and.reduce(conditions)
    return X[idx, :]


def slice_sphere(X, axis=None, rmin=0.0, rmax=None):
    """Return points within a spherical shell slice.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the sphere.
    rmin, rmax : float
        Inner/outer radius of spherical shell.

    Returns
    -------
    ndarray, shape (?, d)
        The points within the sphere.
    """
    if rmax is None:
        rmax = np.inf
    radii = get_radii(project(X, axis))
    idx = np.logical_and(radii > rmin, radii < rmax)
    return X[idx, :]


def slice_ellipsoid(X, axis=None, rmin=0.0, rmax=None):
    """Return points within an ellipsoidal shell slice.

    The ellipsoid is defined by the covariance matrix of the
    distribution.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the ellipsoid.
    rmin, rmax : list[float]
        Min/max "radius" (x^T Sigma^-1 x). relative to covariance matrix.

    Returns
    -------
    ndarray, shape (?, d)
        Points within the shell.
    """
    if rmax is None:
        rmax = np.inf
    radii = get_ellipsoid_radii(project(X, axis))
    idx = np.logical_and(rmin < radii, radii < rmax)
    return X[idx, :]


def slice_contour(X, axis=None, lmin=0.0, lmax=1.0, interp=True, **hist_kws):
    """Return points within a contour shell slice.

    The slice is defined by the density contours in the subspace defined by
    `axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the density contours.
    lmin, lmax : list[float]
        If `f` is the density in the subspace defined by `axis`, then we select
        points where lmin <= f / max(f) <= lmax.
    interp : bool
        If True, compute the histogram, then interpolate and evaluate the
        resulting function at each point in `X`. Otherwise we keep track
        of the indices in which each point lands when it is binned,
        and accept the point if it's bin has a value within fmin and fmax.
        The latter is a lot slower.

    Returns
    -------
    ndarray, shape (?, d)
        Points within the shell.
    """
    _X = project(X, axis)
    hist, edges = histogram(_X, **hist_kws)
    hist = hist / np.max(hist)
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    if interp:
        fint = scipy.interpolate.RegularGridInterpolator(
            centers,
            hist,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        values = fint(_X)
        idx = np.logical_and(lmin <= values, values <= lmax)
    else:
        valid_indices = np.vstack(
            np.where(np.logical_and(lmin <= hist, hist <= lmax))
        ).T
        indices = np.vstack(
            [np.digitize(_X[:, k], edges[k]) for k in range(_X.shape[1])]
        ).T
        idx = []
        for i in range(len(indices)):
            if indices[i].tolist() in valid_indices.tolist():
                idx.append(i)
    return X[idx, :]


def norm_xxp_yyp_zzp(X, scale_emittance=False):
    """Normalize x-px, y-py, z-pz, ...

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional phase space (d is even).
    scale_emittance : bool
        Whether to divide the coordinates by the square root of the rms emittance.

    Returns
    -------
    Xn : ndarray, shape (n, d)
        Normalized phase space coordinate array.
    """
    if X.shape[1] % 2 != 0:
        raise ValueError("X must have an even number of columns.")
    Sigma = np.cov(X.T)
    Xn = np.zeros(X.shape)
    for i in range(0, X.shape[1], 2):
        sigma = Sigma[i : i + 2, i : i + 2]
        alpha, beta = ap.twiss(sigma)
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        if scale_emittance:
            eps = ap.apparent_emittance(sigma)
            Xn[:, i : i + 2] = Xn[:, i : i + 2] / np.sqrt(eps)
    return Xn


def decorrelate(X):
    """Remove cross-plane correlations by permuting (x, x'), (y, y'), (z, z') pairs.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of k points in d-dimensional space (d is even).

    Returns
    -------
    ndarray, shape (n, d)
        The decorrelated coordinates.
    """
    if X.shape[1] % 2 != 0:
        raise ValueError("X must have even number of columns.")
    for i in range(0, X.shape[1], 2):
        idx = np.random.permutation(np.arange(X.shape[0]))
        X[:, i : i + 2] = X[idx, i : i + 2]
    return X


def downsample(X, samples):
    """Select a random subset of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    samples : int or float
        The number of samples to keep If less than 1, specifies
        the fraction of points.

    Returns
    -------
    ndarray, shape (<= n, d)
        The downsampled coordinate array.
    """
    samples = min(samples, X.shape[0])
    idx = random_selection(np.arange(X.shape[0]), samples)
    return X[idx, :]


# Density estimation
# ------------------------------------------------------------------------------


def histogram_bin_edges(X, bins=10, limits=None):
    """Multi-dimensional histogram bin edges.

    This function calls `np.histogram_bin_edges` along each axis of X.

    See https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html

    Parameters
    ----------
    bins : int or str
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10, by default).

        If `bins` is a string, `histogram_bin_edges` will use the method chosen
        to calculate the optimal number of bins.

        if `bins` is a sequence of floats, it defines the bin edges, including
        the rightmost edge.

        A list of {str / int / float sequence} may be provided such that bins[i]
        corresponds to axis i.
    limits : (float, float)
        The lower and upper range of the bins.  If not provided, the limits are
        ``[(np.min(X[:, i]), np.max(X[:, i])) for i in range(X.shape[1])]``.

    Returns
    -------
    edges : list[ndarray]
        Bin edges along each axis.
    """
    if X.ndim == 1:
        return np.histogram_bin_edges(X, bins, limits)
    # `[2, 3, 4, 5]` could mean "2 bins along axis 0, 3 bins along axis 1, ..."
    # or "bin edges [2.0, 3.0, 4.0, 5.0] along each axis". We assume the
    # former if `bins` is a sequence of int and the latter if `bins` is a
    # sequence of float.
    if array_like(bins) and type(bins[0]) is float:
        bins = X.shape[1] * [bins]
    # If a single int/str is provided, apply to all axes.
    if not array_like(bins):
        bins = X.shape[1] * [bins]
    # Same for `limits`. If a (min, max) tuple (or None) is provided, apply
    # to all axes.
    if limits is None or (limits[0] is not None and not array_like(limits[0])):
        limits = X.shape[1] * [limits]
    return [
        np.histogram_bin_edges(X[:, i], bins[i], limits[i]) for i in range(X.shape[1])
    ]


def histogram(X, bins=10, limits=None, centers=False):
    """Multi-dimensional histogram.

    Parameters
    ----------
    See `histogram_bin_edges`.

    Returns
    -------
    See `np.histogramdd`.
    """
    if X.ndim == 1:
        bins = np.histogram_bin_edges(X, bins, limits)
        hist, _ = np.histogram(X, bins=bins)
        if centers:
            bins = utils.centers_from_edges(bins)
        return hist, bins

    bins = histogram_bin_edges(X, bins=bins, limits=limits)
    hist, _ = np.histogramdd(X, bins)
    if centers:
        bins = [utils.centers_from_edges(b) for b in bins]
    return hist, bins


def sparse_histogram(X, bins=10, limits=None, centers=False, eps=1.0e-12):
    """Compute sparse multidimensional histogram.

    Parameters
    ----------
    Same as `histogram`.
    eps : float
        Small constant added to largest bin edge.

    Returns
    ------
    indices : ndarray, shape (k, d)
        Indices of nonzero bins in d-dimensional histogram.
    counts : ndarray, shape (k,)
        Counts of nonzero bins in d-dimensional histogram.
    bins : list(ndarray)
        List of bin edges or centers along each axis.
    """
    bins = histogram_bin_edges(X, bins=bins, limits=limits)
    shape = [len(bins[axis]) for axis in range(X.shape[1])]
    for axis in range(len(bins)):
        bins[axis][-1] = bins[axis][-1] + eps
    # Get multidimensional bin index of each point.
    indices = []
    valid = np.full(X.shape[0], True)
    for axis in range(X.shape[1]):
        idx = np.digitize(X[:, axis], bins[axis])
        valid = np.logical_and(valid, np.logical_and(idx > 0, idx < len(bins[axis])))
        idx = idx - 1  # 0 indexes first bin
        indices.append(idx)
    for axis in range(X.shape[1]):
        indices[axis] = indices[axis][valid]
    # Convert to flat indices.
    shape = [len(bins[axis]) for axis in range(X.shape[1])]
    indices = np.ravel_multi_index(indices, shape)
    # Count the indices/counts of each nonzero bin.
    counter = collections.Counter(indices)
    counts = np.array(list(counter.values()))
    indices = np.array(list(counter.keys()))
    # Convert to multidimensional indices.
    indices = np.unravel_index(indices, shape)
    indices = np.vstack(indices).T
    if centers:
        bins = [centers_from_edges(bins[axis]) for axis in range(X.shape[1])]
    return indices, counts, bins


def gaussian_kde(X, **kws):
    """Gaussian kernel density estimation (KDE).

    This function just calls `scipy.stats.gaussian_kde`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    **kws
        Key word arguments

    Returns
    -------
    estimator : scipy.stats.gaussian_kde
        The density estimator.
    """
    return scipy.stats.gaussian_kde(X.T, **kws)


def radial_histogram(X, **kws):
    """Count number of points within spherical shells, with counts normalized by shell volume.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    **kws
        Key word arguments for `histogram`.
    """
    radii = get_radii(X)
    hist, bins = histogram(radii, **kws)
    if "centers" in kws and kws["centers"]:
        _edges = utils.edges_from_centers(bins)
    else:
        _edges = bins
    for i in range(len(_edges) - 1):
        rmin = _edges[i]
        rmax = _edges[i + 1]
        hist[i] = hist[i] / utils.volume_sphere_shell(
            rmin=rmin, rmax=rmax, n=X.shape[1]
        )
    return hist, bins
