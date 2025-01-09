from typing import Callable
from typing import Self

import numpy as np
from scipy import ndimage

from .cov import cov_to_corr
from .points import covariance_matrix
from .utils import array_like
from .utils import coords_to_edges
from .utils import edges_to_coords


class GridBase:
    def __init__(self, coords: np.ndarray = None, edges: np.ndarray = None) -> None:
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
        self.cell_volume = np.prod([c[1] - c[0] for c in self.coords])

        self.values = None

    def normalize(self) -> None:
        values_sum = np.sum(self.values)
        if values_sum > 0.0:
            self.values = self.values / values_sum / self.cell_volume

    def mesh(self) -> list[np.ndarray]:
        return np.meshgrid(*self.coords, indexing='ij')

    def points(self) -> np.ndarray:
        return np.vstack([C.ravel() for C in np.meshgrid(*self.coords, indexing='ij')]).T


class Grid(GridBase):
    def __init__(
        self,
        values: np.ndarray = None,
        coords: np.ndarray = None,
        edges: np.ndarray = None,
    ) -> None:
        super().__init__(coords=coords, edges=edges)

        self.values = values
        if self.values is None:
            self.values = np.zeros(self.shape)

    def max_indices(self) -> tuple[np.ndarray]:
        return np.unravel_index(np.argmax(self.values), self.shape)

    def sample(self, size: int, noise: float = 0.0) -> np.ndarray:
        return sample_grid(size=size, noise=noise)


class SparseGrid(BaseGrid):
    def __init__(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        coords: np.ndarray = None,
        edges: np.ndarray = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        values : ndarray, shape (K,)
            Counts in each bin. Does not need to be normalized.
        indices : ndarray, shape (K, N)
            Indices of nonzero cells in N-dimensional grid.
        """
        super().__init__(coords=coords, edges=edges)
        self.values = values
        self.indices = indices

    def sample(self, size: int, noise: float = 0.0) -> np.ndarray:
        return sample_sparse_grid(size=size, noise=noise)


def slice_idx(
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


def slice_idx_ellipsoid(
    grid: Grid,
    axis: tuple[int, ...],
    covariance_matrix: np.ndarray,
    rmin: float,
    rmax: float,
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


def slice_idx_contour(
    grid: Grid,
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


def _slice(
    grid: Grid,
    axis: int | tuple[int, ...],
    ind: int | tuple[int, ...] | list[tuple[int, ...]],
    return_indices: bool = False,
) -> GridDist:

    ndim = grid.ndim
    idx = grid_slice_idx(ndim=grid.ndim, axis=axis, ind=ind)

    coords_new = []
    for axis in range(ndim):
        ilo, ihi = idx[axis].start
        if ilo is None:
            ilo = 0

        ihi = idx[axis].stop
        if ihi is None:
            ihi = grid.shape[axis]

        if ihi - ilo > 1:
            coords_new.append(coords[axis][idx[axis]])

    values_new = grid.values[idx]

    grid_new = None
    if values_new.ndim == 1:
        grid_new = GridDist1D(values=values_new, coords=coords_new)
    else:
        grid_new = GridDist(values=values_new, coords=coords_new)

    if return_indices:
        return grid_new, idx
    return grid_new


def _slice_ellipsoid(
    grid: Grid,
    axis: tuple[int, ...],
    rmin: float,
    rmax: float,
    return_indices: bool = False,
) -> GridDist:

    idx = slice_idx_ellipsoid(
        axis=axis,
        covariance_matrix=covariance_matrix(grid),
        rmin=rmin,
        rmax=rmax,
    )

    values_new = grid.values[idx]
    coords_new = [grid.coords[i] for i in range(grid.ndim) if i not in axis]

    grid_new = GridDist(values=values_new, coords=coords_new)
    if return_indices:
        return grid_new, idx
    return grid_new

def _slice_contour(
    grid: Grid,
    axis: tuple[int, ...],
    lmin: float = 0.0,
    lmax: float = 1.0,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

    idx = slice_idx_contour(grid, axis=axis, lmin=lmin, lmax=lmax)

    values_new = grid.values[idx]
    coords_new = [grid.coords[i] for i in range(grid.ndim) if i not in axis]

    grid_new = GridDist(values=values_new, coords=coords_new)
    if return_indices:
        return grid_new, idx
    return grid_new

def project(grid: Grid, axis: int | tuple[int, ...]) -> GridDist:
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
    grid: Grid,
    axis: int = 0,
    lmin: float = 0.0,
    lmax: float = 1.0,
    grid_proj: GridDist = None,
) -> GridDist:
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
        The 1D projection of the (N-1)D slice.
    """
    coords = grid.coords
    values = grid.values

    axis_proj = [i for i in range(grid.ndim) if i != axis]
    if grid_proj is None:
        grid_proj = project(grid, axis=axis_proj)
    values_proj = grid_proj.values / np.max(grid_proj.values)

    grid_slice = _slice(
        grid,
        axis=axis_proj,
        ind=np.where(np.logical_and(values_proj >= lmin, values_proj <= lmax)),
    )

    # `values[idx]` will give a two-dimensional array. Normally we need to sum over
    # the first axis. If `axis == 0`, we need to sum over the second axis.

    values_slice = grid_slice.values
    values_proj_1d = np.sum(values_slice, axis=int(axis == 0))
    coords_proj_1d = grid.coords[axis]
    return Grid1D(values=values_proj_1d, coords=coords_proj_1d)


def project_contour_slice_2d(
    grid: Grid,
    axis: tuple[int, int] = (0, 1),
    lmin: float = 0.0,
    lmax: float = 1.0,
    grid_proj: GridDist = None,
) -> np.ndarray:
    """Apply contour slice in n - 2 dimensions, then project onto the remaining two dimensions.

    Parameters
    ----------
    grid : GridDist
        Distribution on N-dimensional grid.
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
    GridDist
        The 2D projection of the (N-2)D slice.
    """
    axis_proj = [k for k in range(values.ndim) if k not in axis]
    axis_proj = tuple(axis_proj)

    if grid_proj is None:
        grid_proj = project(grid, axis=axis_proj)

    values_proj = grid_proj.values / np.max(grid_proj.values)

    grid_slice = _slice(
        grid=grid,
        axis=axis_proj,
        ind=np.where(np.logical_and(values_proj >= lmin, values_proj <= lmax)),
    )

    # `values[idx]` will give a three-dimensional array. Normally we need to sum over
    # the first axis. If `axis == (0, 1)`, we need to sum over the third axis.
    # If `axis == (0, n - 1), we need to sum over the second axis.
    _axis_proj = (1, 2)
    if axis == (0, 1):
        _axis_proj = (0, 1)
    elif axis == (0, grid.ndim - 1):
        _axis_proj = (0, 2)

    # Two elements of `idx` will be `slice(None)`; these are the elements in `axis`.
    # These will always be in order. So, if `axis[0] > axis[1]`, we need to flip
    # `axis_proj`. Need a way to handle this automatically.
    if axis[0] > axis[1]:
        _axis_proj = tuple(reversed(_axis_proj))

    return project(grid_slice, axis=_axis_proj)


def copy_values_into_new_dim(
    values: np.ndarray,
    shape: tuple[int, ...],
    axis: int = -1,
    method: str = "broadcast",
    copy: bool = False,
) -> np.ndarray:
    """Copy image into one or more new dimensions.

    https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times

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


def sample_grid(grid: Grid, size: int = 100, noise: float = 0.0) -> np.ndarray:

    values = grid.values
    edges = grid.edges

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


def sample_sparse_grid(grid: SparseGridDist, size: int = 100, noise: float = 0.0) -> np.ndarray:
    """Sample from sparse histogram.

    Parameters
    ----------
    grid : SparseGridDist
        Sparse N-dimensional grid.
    size : int
        The number of samples to draw.
    noise : float
        Add noise to each particle; a number is drawn uniformly from a box centered on the
        particle with dimensions equal to the histogram bin dimensions. `noise` scales the
        box dimensions relative to the bin dimensions.

    Returns
    -------
    ndarray, shape (size, N)
        Samples drawn from the distribution.
    """
    values = grid.values
    indices = grid.indices
    edges = grid.edges

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


def mean(self) -> np.ndarray:
    x = [np.average(C, weights=self.values) for C in self.mesh()]
    x = np.array(x)
    return x


def cov(grid: Grid) -> np.ndarray:
    def cov_2d(values: np.ndarray, coords: list[np.ndarray]) -> np.ndarray:
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
        return cov_2d(grid.values, grid.coords)

    S = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i):
            # Compute 2 x 2 covariance matrix from this projection.
            axis = (i, j)
            grid_proj = project(grid, axis)
            S_proj = cov_2d(grid_proj.values, grid_proj.coords)

            # Update elements of n x n covariance matrix. This will update
            # some elements multiple times, but it should not matter.
            S[i, i] = S_proj[0, 0]
            S[j, j] = S_proj[1, 1]
            S[i, j] = S[j, i] = S_proj[0, 1]
    return S


def corr(grid: Grid) -> np.ndarray:
    return cov_to_corr(cov(grid))


def ellipsoid_radii(grid: Grid, covariance_matrix: np.ndarray) -> np.ndarray:
    """Return covariance ellipsoid radii from grid coordinates and covariance matrix.

    Radius is defined as x^T Sigma^-1^T. This function computes the radius
    at every point on the grid.

    This is quite slow when n > 4.

    Parameters
    ----------
    grid: Grid
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
    grid: Grid,
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



# Processing
# --------------------------------------------------------------------------------------

def blur(grid: Grid, sigma: float) -> np.ndarray:
    """Call scipy.ndimage.gaussian_filter."""
    grid.values = ndimage.gaussian_filter(grid.values, sigma)
    return grid


def clip(grid: Grid, lmin: float = None, lmax: float = None, frac: bool = False) -> np.ndarray:
    """Clip between lmin and lmax, can be fractions or absolute values."""
    if not (lmin or lmax):
        return grid
    if frac:
        values = grid.values
        values_max = np.max(values)
        if lmin:
            lmin = values_max * lmin
        if lmax:
            lmax = values_max * lmax
    grid.values = np.clip(values, lmin, lmax)
    return grid


def fill(grid: np.ndarray, fill_value: float = None) -> np.ndarray:
    """Call numpy.ma.filled."""
    grid.values = np.ma.filled(grid.values, fill_value=fill_value)
    return grid


def thresh(grid: Grid, lmin: float = None, frac: bool = False) -> np.ndarray:
    if lmin:
        if frac:
            lmin = lmin * grid.values.max()
        grid.values[grid.values < lmin] = 0.0
    return grid
