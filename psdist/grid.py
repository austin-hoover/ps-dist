from typing import Callable
from typing import Self

import numpy as np
from scipy import ndimage

from .cov import cov_to_corr
from .points import covariance_matrix
from .utils import array_like
from .utils import coords_to_edges
from .utils import edges_to_coords


class Grid1D:
    def __init__(self, coords: np.ndarray = None, edges: np.ndarray = None) -> None:
        self.coords = coords
        self.edges = edges
        
        if (self.coords is None) and (self.edges is not None):
            self.coords = edges_to_coords(self.edges)
        
        if (self.edges is None) and (self.coords is not None):
            self.edges = coords_to_edges(self.coords)

        self.shape = (len(self.coords),)
        self.size = len(self.coords)
        self.ndim = 1


class Grid:
    def __init__(self, coords: np.ndarray = None, edges: np.ndarray = None) -> None:
        self.coords = coords
        self.edges = edges
        
        if (self.coords is None) and (self.edges is not None):
            self.coords = edges_to_coords(self.edges)
        
        if (self.edges is None) and (self.coords is not None):
            self.edges = coords_to_edges(self.coords)
            
        if type(self.coords) not in [tuple, list]:
            self.coords = [self.coords]

        if type(self.edges) not in [tuple, list]:
            self.edges = [self.edges]

        self.shape = ([len(c) for c in self.coords])
        self.size = np.prod(self.shape)
        self.ndim = len(self.shape)


class GridDist(Grid):
    def __init__(self, values: np.ndarray = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.values = values
        if self.values is None:
            self.values = np.zeros(shape)


class GridDist1D(Grid):
    def __init__(self, values: np.ndarray = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.values = values
        if self.values is None:
            self.values = np.zeros(shape)


class SparseGrid:
    def __init__(self):
        raise NotImplementedError


class SparseGridDist(SparseGrid):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


def meshpoints(grid: GridDist) -> np.ndarray:
    coords = grid.coords
    return np.vstack([C.ravel() for C in np.meshgrid(*coords, indexing="ij")]).T


def max_indices(grid: GridDist) -> tuple[np.ndarray]:
    """Return the indices of the maximum element."""
    values = grid.values
    return np.unravel_index(np.argmax(values), values.shape)


def ellipsoid_radii(grid: GridDist, covariance_matrix: np.ndarray) -> np.ndarray:
    """Return covariance ellipsoid radii from grid coordinates and covariance matrix.

    Radius is defined as x^T Sigma^-1^T. This function computes the radius
    at every point on the grid.

    This is quite slow when n > 4.

    Parameters
    ----------
    grid: GridDist
        Distribution on N-dimensional grid.
    covariance_matrix: np.ndarray
        N x N covariance matrix.

    Returns
    -------
    ndarray: np.ndarray
        Radius x^T Sigma^-1^T x at each point in grid.
    """
    shape = grid.shape
    coords = grid.coords
    COORDS = np.meshgrid(*coords, indexing="ij")

    S = covariance_matrix
    S_inv = np.linalg.inv(S)

    R = np.zeros(shape)
    for ii in np.ndindex(shape):
        v = np.array([C[ii] for C in COORDS])
        R[ii] = np.sqrt(np.linalg.multi_dot([v.T, S_inv, v]))
    return R


def radial_density(
    grid: GridDist,
    covariance_matrix: np.ndarray,
    radii: np.ndarray,
    dr: float = None,
) -> np.ndarray:
    """Return average density within ellipsoidal shells.

    Parameters
    ----------
    values : ndarray
        An n-dimensional image.
    R : ndarray, same shape as `values`.
        Gives the radius at each point in `values`.
    radii : ndarray, shape (k,)
        Radii at which to evaluate the density.
    dr : float
        The radial shell width.

    Returns
    -------
    ndarray, shape (k,)
        The average density within each ellipsoidal shell.
    """
    values = grid.values

    R = ellipsoid_radii(grid, covariance_matrix)
    if dr is None:
        dr = 0.5 * np.max(R) / (len(R) - 1)

    values_r = []
    for r in radii:
        values_masked = np.ma.masked_where(np.logical_or(R < r, R > r + dr), values)
        values_r.append(np.mean(values_masked))
    return np.array(values_r)


def mean(grid: GridDist) -> np.ndarray:
    coords = grid.coords
    values = grid.values
    x = [np.average(C, weights=values) for C in np.meshgrid(*coords, indexing="ij")]
    x = np.array(x)
    return x


def covariance_matrix(grid: GridDist) -> np.ndarray:
    def covariance_matrix_2d(values: np.ndarray, coords: list[np.ndarray]) -> np.ndarray:
        COORDS = np.meshgrid(*coords, indexing="ij")
        S = np.zeros((values.ndim, values.ndim))
        values_sum = np.sum(values)
        if values_sum > 0:
            mean = np.array([np.average(C, weights=values) for C in COORDS])
            for i in range(values.ndim):
                for j in range(i + 1):
                    X = COORDS[i] - mean[i]
                    Y = COORDS[j] - mean[j]
                    EX = np.sum(values * X) / values_sum
                    EY = np.sum(values * Y) / values_sum
                    EXY = np.sum(values * X * Y) / values_sum
                    S[i, j] = S[j, i] = EXY - EX * EY
        return S

    ndim = grid.ndim

    if ndim < 3:
        return covariance_matrix_2d(grid.values, grid.coords)

    S = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i):
            # Compute 2 x 2 covariance matrix from this projection.
            axis = (i, j)
            grid_proj = project(grid, axis)
            S_proj = covariance_matrix_2d(grid_proj.values, grid_proj.coords)

            # Update elements of n x n covariance matrix. This will update
            # some elements multiple times, but it should not matter.
            S[i, i] = S_proj[0, 0]
            S[j, j] = S_proj[1, 1]
            S[i, j] = S[j, i] = S_proj[0, 1]
    return S


def correlation_matrix(grid: GridDist) -> np.ndarray:
    return cov_to_corr(covariance_matrix(grid))


def expected_value(grid: GridDist, function: Callable) -> float:
    coords = grid.coords
    values = grid.values

    pdf = np.copy(values) / np.sum(values)
    pdf_flat = pdf.ravel()

    value = 0.0
    for i, x in enumerate(meshpoints(grid)):
        value += function(x) * pdf_flat[i]
    return value


def moment(grid: GridDist, axis: tuple[int, ...], order: tuple[int, ...]) -> float:
    function = lambda x: np.prod([x[k] ** order[i] for i, k in enumerate(axis)])
    return expected_value(grid, function)


def halo_parameter(grid: GridDist) -> float:
    q2 = moment(grid, axis=(0,), order=(2,))
    p2 = moment(grid, axis=(1,), order=(2,))
    q4 = moment(grid, axis=(0,), order=(4,))
    p4 = moment(grid, axis=(1,), order=(4,))
    qp = moment(grid, axis=(0, 1), order=(1, 1),)
    q2p2 = moment(grid, axis=(0, 1), order=(2, 2))
    qp3 = moment(grid, axis=(0, 1), order=(1, 3))
    q3p = moment(grid, axis=(0, 1), order=(3, 1))

    numer = np.sqrt(3.0 * q4 * p4 + 9.0 * (q2p2**2) - 12.0 * qp3 * q3p)
    denom = 2.0 * q2 * p2 - 2.0 * (qp**2)
    return (numer / denom) - 2.0


def get_slice_idx(
    ndim: int,
    axis: int | tuple[int, ...],
    ind: int | tuple[int, ...] | list[tuple[int, ...]]
) -> tuple[slice, ...]:
    """Return planar slice index array.

    Parameters
    ----------
    ndim : int
        The number of elements in the slice index array. (The number of dimensions in the array to be sliced.)
    axis : int or tuple[int, ...]
        The sliced axes.
    ind : int, tuple[int, ...] or list[tuple[int, ...]]
        The indices along the sliced axes. If a tuple is provided, this defines the (min, max) index.

    Returns
    -------
    idx : tuple
        The slice index array. A slice of the array `values` may then be accessed as `values[idx]`.
    """
    # Make list if only one axis provided.
    if type(axis) is int:
        axis = [axis]
        # Can also provide only one axis but provide a tuple for ind, which
        # selects a range along that axis.
        if type(ind) is tuple:
            ind = [ind]

    # Make list if only one ind provided.
    if type(ind) is int:
        ind = [ind]

    # Initialize the slice index to select all elements.
    idx = ndim * [slice(None)]

    # If any indices were provided, add them to `idx`.
    for k, item in zip(axis, ind):
        if item is None:
            continue
        elif (type(item) is tuple) and (len(item) == 2):
            idx[k] = slice(item[0], item[1])
        else:
            # Could be int or list of ints
            idx[k] = item

    return tuple(idx)


def get_slice_idx_ellipsoid(
    axis: tuple[int, ...],
    covariance_matrix: np.ndarray,
    rmin: float,
    rmax: float
) -> tuple[slice, ...]:
    """Compute an ellipsoid slice from covariance matrix.

    Parameters
    ----------
    values : ndarray
        An n-dimensional image.
    axis : tuple[int, ...]
        Specificies the subspace in which the ellipsoid slices are computed.
        Example: in x-y-z space, we may define a circle in x-y. This could
        select points within a cylinder in x-y-z.
    rmin, rmax : float
        We select the region between two nested ellipsoids with "radius"
        rmin and rmax. The radius is r = x^T Sigma^-1 x, where Sigma is
        the covariance matrix and x is the coordinate vector. r = 1 is
        the covariance ellipsoid.

    Returns
    -------
    np.ma.masked_array
        A version of `values` in which elements outside the slice are masked.
    """
    # Will need to compute an (N - M)-dimensional mask (M = len(axis)), then
    # copy the mask into the remaining dimensions with `copy_into_new_dim`.
    raise NotImplementedError


def get_slice_idx_contour(
    grid: GridDist,
    axis: tuple[int, ...],
    lmin: float = 0.0,
    lmax: float = 1.0,
) -> tuple[slice, ...]:
    """Compute a contour slice.

    Parameters
    ----------
    grid : GridDist
        Distribution on N-dimensional grid.
    axis : tuple[int, ...]
        Specificies the subspace in which the contours are computed. (See
        `slice_idx_ellipsoid`.)
    lmin, lmax : float
        `values`is projected onto `axis` and the projection `values_proj` is normalized to
        the range [0, 1]. Then, we find the points in this subspace such that
        `values_proj` is within the range [lmin, lmax].

    Returns
    -------
    np.ma.masked_array
        A version of `values` in which elements outside the slice are masked.
    """
    # Will need to compute an (N - M)-dimensional mask (M = len(axis)), then
    # copy the mask into the remaining dimensions with `copy_into_new_dim`.
    raise NotImplementedError


def slice_grid(
    values: np.ndarray,
    axis: Union[int, tuple[int, ...]],
    ind: Union[int, tuple[int, ...], list[tuple[int, ...]]],
) -> np.ndarray:
    """Return values[idx] for"""
    idx = slice_idx(values.ndim, axis=axis, ind=ind)
    return values[idx]


def _slice_ellipsoid(values: np.ndarray, axis: tuple[int, ...], rmin: float, rmax: float) -> np.ndarray:
    idx = slice_idx_ellipsoid(values, axis=axis, rmin=rmin, rmax=rmax)
    return values[idx]


def _slice_contour(
    values: np.ndarray, axis: tuple[int, ...], lmin: float = 0.0, lmax: float = 1.0
) -> np.ndarray:
    idx = slice_idx_contour(values, axis=axis, lmin=lmin, lmax=lmax)
    return values[idx]


def project(grid: GridDist, axis: int | tuple[int, ...]) -> GridDist:
    """Project grid distribution onto axis.

    Parameters
    ----------
    grid : GridDist
        Distribution on N-dimensional grid.
    axis : int | tuple[int, ...]
        The axes onto which the image is projected, i.e., the axes which are not summed over.
        Array axes are swapped as required.

    Returns
    -------
    GridDist
        The projection of `grid` onto the specified axis.
    """
    if grid.ndim == 1:
        return grid

    coords = grid.coords
    values = grid.values

    # Sum over specified axes.
    if type(axis) is int:
        axis = [axis]
    axis = tuple(axis)
    axis_sum = tuple([i for i in range(values.ndim) if i not in axis])
    values_proj = np.sum(values, axis_sum)

    # Order the remaining axes.
    loc = list(range(proj.ndim))
    destination = np.zeros(proj.ndim, dtype=int)
    for i, index in enumerate(np.argsort(axis)):
        destination[index] = i
    for i in range(proj.ndim):
        if loc[i] != destination[i]:
            j = loc.index(destination[i])
            values_proj = np.swapaxes(proj, i, j)
            (loc[i], loc[j]) = (loc[j], loc[i])

    # Make new grid
    coords_proj = [grid.coords[i] for i in axis]
    grid_proj = GridDist(values=values_proj, coords=coords_proj)
    return grid_proj


def project_contour_slice_1d(
    grid: GridDist,
    axis: int = 0,
    lmin: float = 0.0,
    lmax: float = 1.0,
    grid_proj: GridDist = None,
) -> np.ndarray:
    """Apply contour slice in N- 1 dimensions, then project onto the remaining dimension.

    Parameters
    ----------
    grid : GridDist
        Distribution on N-dimensional grid.
    axis : int
        The projection axis.
    lmin, lmax : float
        Min and max contour levels of the (N-1)-dimensional projection of `values`,
        normalized the range [0, 1].
    grid_proj : GridDist
        The (N-1)-dimensional projection of `grid` onto all dimensions other than `axis`.
        If not provided, it will be computed within the function.

        Shape:
        ```
        tuple([grid.shape[i] for i in range(grid.ndim) if i != axis])
        ```

    Returns
    -------
    GridDist
        The projection of the sliced grid.
    """
    axis_proj = [i for i in range(values.ndim) if i != axis]
    if values_proj is None:
        values_proj = project(values, axis=axis_proj)
    values_proj = values_proj / np.max(values_proj)
    idx = slice_idx(
        ndim=values.ndim,
        axis=axis_proj,
        ind=np.where(np.logical_and(values_proj >= lmin, values_proj <= lmax)),
    )
    # `f[idx]` will give a two-dimensional array. Normally we need to sum over
    # the first axis. If `axis == 0`, we need to sum over the second axis.
    return np.sum(values[idx], axis=int(axis == 0))


def project2d_contour(
    values: np.ndarray,
    axis: tuple[int, ...] = (0, 1),
    lmin: float = 0.0,
    lmax: float = 1.0,
    values_proj: np.ndarray = None,
) -> np.ndarray:
    """Apply contour slice in n - 2 dimensions, then project onto the remaining two dimensions.

    Parameters
    ----------
    values : ndarray
        An n-dimensional image.
    axis : tuple
        The 2D projection axis.
    lmin, lmax : float
        Min and max contour levels of the (n-2)-dimensional projection of `values`,
        normalized the the range [0, 1].
    values_proj : ndarray, shape = [values.shape[i] for i in range(values.ndim) if i != axis]
        The (n-2)-dimensional projection of `values` onto all dimensions other than `axis`.
        (If not provided, it will be computed within the function.)

    Returns
    -------
    ndarray, shape = (values.shape[i] for i in axis]
        The 2D projection of the slice.
    """
    axis_proj = [k for k in range(values.ndim) if k not in axis]
    axis_proj = tuple(axis_proj)

    if values_proj is None:
        values_proj = project(values, axis=axis_proj)

    values_proj = values_proj / np.max(values_proj)

    idx = slice_idx(
        values.ndim,
        axis=axis_proj,
        ind=np.where(np.logical_and(values_proj >= lmin, values_proj <= lmax)),
    )

    # `values[idx]` will give a three-dimensional array. Normally we need to sum over
    # the first axis. If `axis == (0, 1)`, we need to sum over the third axis.
    # If `axis == (0, n - 1), we need to sum over the second axis.
    _axis_proj = (1, 2)
    if axis == (0, 1):
        _axis_proj = (0, 1)
    elif axis == (0, values.ndim - 1):
        _axis_proj = (0, 2)

    # Two elements of `idx` will be `slice(None)`; these are the elements in `axis`.
    # These will always be in order. So, if `axis[0] > axis[1]`, we need to flip
    # `axis_proj`. Need a way to handle this automatically.
    if axis[0] > axis[1]:
        _axis_proj = tuple(reversed(_axis_proj))
    return project(values[idx], axis=_axis_proj)


def copy_into_new_dim(
    values: np.ndarray,
    shape: tuple[int, ...],
    axis: int = -1,
    method: str = "broadcast",
    copy: bool = False,
) -> np.ndarray:
    """Copy image into one or more new dimensions.

    See [https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times]

    Parameters
    ----------
    values : ndarray
        An n-dimensional image.
    shape : d-tuple of ints
        The shape of the new dimensions.
    axis : int (0 or -1)
        If 0, the new dimensions will be inserted before the first axis. If -1,
        the new dimensions will be inserted after the last axis. I think
        values other than 0 or -1 should work; this does not currently
        work, at least for `method='broadcast'`, last I checked.
    method : {'repeat', 'broadcast'}
        Whether to use `np.repeat` or `np.expand_dims` and `np.broadcast_to`. The
        'broadcast' method is faster.
    """
    if not array_like(shape):
        shape = (shape,)

    if method == "repeat":
        for i in range(len(shape)):
            values = np.repeat(np.expand_dims(values, axis), shape[i], axis=axis)
        return values
    elif method == "broadcast":
        if axis == 0:
            new_shape = shape + values.shape
        elif axis == -1:
            new_shape = values.shape + shape
        else:
            raise ValueError("Cannot yet handle axis != 0, -1.")

        for _ in range(len(shape)):
            values = np.expand_dims(values, axis)

        if copy:
            return np.broadcast_to(values, new_shape).copy()
        else:
            return np.broadcast_to(values, new_shape)
    else:
        raise ValueError


# Processing
# --------------------------------------------------------------------------------------


def blur(values: np.ndarray, sigma: float) -> np.ndarray:
    """Call scipy.ndimage.gaussian_filter."""
    return ndimage.gaussian_filter(values, sigma)


def clip(
    values: np.ndarray, lmin: float = None, lmax: float = None, frac: bool = False
) -> np.ndarray:
    """Clip between lmin and lmax, can be fractions or absolute values."""
    if not (lmin or lmax):
        return values
    if frac:
        f_max = np.max(f)
        if lmin:
            lmin = f_max * lmin
        if lmax:
            lmax = f_max * lmax
    return np.clip(f, lmin, lmax)


def fill(values: np.ndarray, fill_value: float = None) -> np.ndarray:
    """Call numpy.ma.filled."""
    return np.ma.filled(values, fill_value=fill_value)


def normalize(values: np.ndarray, norm: str = "volume", pixel_volume: float = 1.0) -> np.ndarray:
    """Scale to unit volume or unit maximum."""
    factor = 1.0
    if norm == "volume":
        factor = np.sum(values) * pixel_volume
    elif norm == "max":
        factor = np.max(values)
    if factor == 0.0:
        return values
    return values / factor


def threshold(values: np.ndarray, lmin: float = None, frac: bool = False) -> np.ndarray:
    """Set pixels less than lmin to zero."""
    if lmin:
        if frac:
            values_max = np.max(values)
            lmin = lmin * values_max
        values[values < lmin] = 0.0
    return values


# Sampling
# --------------------------------------------------------------------------------------


def sample(
    values: np.ndarray,
    edges: list[np.ndarray] = None,
    coords: list[np.ndarray] = None,
    size: int = 100,
    noise: float = 0.0,
) -> np.ndarray:
    """Sample particles from image.

    Parameters
    ----------
    values : ndarray
        An n-dimensional image/histogram.
    coords, edges : list[ndarray], length n
        Bin centers/edges along each axis.
    size : int
        The number of samples to draw.
    noise : float
        Uniform random noise for smoothing.

    Returns
    -------
    ndarray, shape (size, n)
        Samples drawn from the distribution.
    """
    if edges is None:
        if coords is None:
            edges = [np.arange(s + 1) for s in values.shape]
        else:
            edges = [edges_from_coords(c) for c in coords]
    elif values.ndim == 1:
        edges = [edges]

    idx = np.flatnonzero(values)
    pdf = values.ravel()[idx]
    pdf = pdf / np.sum(pdf)
    idx = np.random.choice(idx, size, replace=True, p=pdf)
    idx = np.unravel_index(idx, shape=values.shape)
    lb = [edges[axis][idx[axis]] for axis in range(values.ndim)]
    ub = [edges[axis][idx[axis] + 1] for axis in range(values.ndim)]

    points = np.squeeze(np.random.uniform(lb, ub).T)
    if noise:
        for axis in range(points.shape[1]):
            delta = ub[axis] - lb[axis]
            points[:, axis] += noise * 0.5 * np.random.uniform(-delta, delta, size=points.shape[0])
    return points


def sample_sparse(
    values: np.ndarray,
    indices: np.ndarray,
    edges: list[np.ndarray] = None,
    coords: list[np.ndarray] = None,
    size: int = 1,
    noise: float = 0.0,
) -> np.ndarray:
    """Sample from sparse histogram.

    Parameters
    ----------
    values : ndarray, shape (k,)
        Counts in each bin. Does not need to be normalized.
    indices : ndarray, shape (k, n)
        Indices of nonzero bins in n-dimensional image/histogram.
    coords, edges : list[ndarray], length n
        Bin centers/edges along each axis.
    size : int
        The number of samples to draw.
    noise : float
        Add noise to each particle; a number is drawn uniformly from a box centered on the
        particle with dimensions equal to the histogram bin dimensions. `noise` scales the
        box dimensions relative to the bin dimensions.

    Returns
    -------
    ndarray, shape (size, n)
        Samples drawn from the distribution.
    """
    if edges is None:
        edges = [edges_from_coords(c) for c in coords]

    shape = [len(e) - 1 for e in edges]
    indices_flat = np.ravel_multi_index(indices.T, shape)
    idx = np.random.choice(indices_flat, size=size, replace=True, p=(values / np.sum(values)))
    idx = np.unravel_index(idx, shape=shape)
    lb = [edges[axis][idx[axis]] for axis in range(len(shape))]
    ub = [edges[axis][idx[axis] + 1] for axis in range(len(shape))]
    points = np.squeeze(np.random.uniform(lb, ub).T)
    if noise:
        delta = 0.5 * noise * np.array([np.mean(np.diff(e)) for e in edges])
        points += np.random.uniform(low=-delta, high=delta, size=points.shape)
    return points


