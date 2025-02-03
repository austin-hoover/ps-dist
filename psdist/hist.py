from typing import Callable
from typing import Self

import numpy as np

from .cov import cov_to_corr
from .utils import array_like


def get_grid_points(coords: list[np.ndarray]) -> np.ndarray:
    return np.vstack([C.ravel() for C in np.meshgrid(*coords, indexing="ij")]).T


def squeeze_coords(coords: list[np.ndarray]) -> np.ndarray:
    if len(coords) == 1:
        return coords[0]
    return coords


def edges_to_coords_1d(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def coords_to_edges_1d(coords: np.ndarray) -> np.ndarray:
    delta = np.diff(coords)[0]
    return np.hstack([coords - 0.5 * delta, [coords[-1] + 0.5 * delta]])


def edges_to_coords(
    edges: np.ndarray | list[np.ndarray],
) -> np.ndarray | list[np.ndarray]:
    """Compute bin center coordinates from evenly spaced bin edges."""
    coords = None
    if np.isscalar(edges[0]):
        coords = edges_to_coords_1d(edges)
    else:
        coords = [edges_to_coords_1d(e) for e in edges]
    return coords


def coords_to_edges(coords: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Compute bin edges from evenly spaced bin coordinates."""
    edges = None
    if np.isscalar(coords[0]):
        edges = coords_to_edges_1d(coords)
    else:
        edges = [coords_to_edges_1d(c) for c in coords]
    return edges


class Grid:
    def __init__(
        self, coords: list[np.ndarray] = None, edges: list[np.ndarray] = None
    ) -> None:
        self.coords = coords
        self.edges = edges

        if (self.coords is None) and (self.edges is not None):
            self.coords = edges_to_coords(self.edges)

        if (self.edges is None) and (self.coords is not None):
            self.edges = coords_to_edges(self.coords)

        self.shape = tuple([len(c) for c in self.coords])
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)
        self.cell_volume = np.prod([c[1] - c[0] for c in self.coords])

        self.limits = [(e[0], e[-1]) for e in self.edges]
        self.limits = np.array(self.limits)

        self.ranges = [(xmax - xmin) for (xmin, xmax) in self.limits]

    def points(self) -> np.ndarray:
        return get_grid_points(self.coords)

    def meshgrid(self) -> np.ndarray:
        return np.meshgrid(*self.coords, indexing="ij")

    def normalize(self) -> None:
        values_sum = np.sum(self.values)
        if values_sum > 0.0:
            self.values = self.values / values_sum / self.cell_volume

    def grid_points(self) -> np.ndarray:
        return get_grid_points(self.coords)


class Histogram(Grid):
    def __init__(
        self,
        values: np.ndarray = None,
        coords: np.ndarray = None,
        edges: np.ndarray = None,
    ) -> None:
        if values is not None:
            if values.ndim == 1:
                raise ValueError("values.ndim < 2")

        super().__init__(coords=coords, edges=edges)

        self.values = values
        if self.values is None:
            self.values = np.zeros(self.shape)

    def copy(self) -> Self:
        return Histogram(values=self.values, edges=self.edges)

    def max_indices(self) -> tuple[np.ndarray]:
        return np.unravel_index(np.argmax(self.values), self.shape)

    def scale_max(self) -> None:
        max_value = np.max(self.values)
        if max_value > 0.0:
            self.values = self.values / max_value

    def sample(self, size: int, noise: float = 0.0) -> np.ndarray:
        return sample_hist(self, size=size, noise=noise)

    def project(self, axis: int | tuple[int, ...]) -> Self:
        return project(self, axis, squeeze=True)

    def slice(
        self,
        axis: int | tuple[int, ...],
        ind: int | tuple[int, ...] | list[tuple[int, ...]],
        return_indices: bool = False,
    ) -> Self | tuple[Self, list[slice]]:
        return slice_(self, axis=axis, ind=ind, return_indices=return_indices)

    def bin(self, points: np.ndarray, density: bool = True) -> np.ndarray:
        self.values, _ = np.histogramdd(points, bins=self.edges, density=density)
        return np.copy(self.values)

    def __call__(self, **kwargs) -> np.ndarray:
        return self.bin(**kwargs)


class SparseHistogram(Grid):
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
        values: ndarray, shape (K,)
            Counts in each bin. Does not need to be normalized.
        indices: ndarray, shape (K, N)
            Indices of nonzero cells in N-dimensional hist.
        """
        super().__init__(coords=coords, edges=edges)
        self.values = values
        self.indices = indices
        self.size = len(self.values)  # number of nonzero cells

    def sample(self, size: int, noise: float = 0.0) -> np.ndarray:
        return sample_sparse_hist(self, size=size, noise=noise)


class Histogram1D:
    def __init__(
        self,
        values: np.ndarray = None,
        coords: np.ndarray = None,
        edges: np.ndarray = None,
    ) -> None:
        self.coords = coords
        self.edges = edges

        if (self.coords is None) and (self.edges is not None):
            self.coords = edges_to_coords(self.edges)

        if (self.edges is None) and (self.coords is not None):
            self.edges = coords_to_edges(self.coords)

        self.shape = (len(self.coords),)
        self.ndim = 1
        self.size = len(self.coords)
        self.cell_volume = self.coords[1] - self.coords[0]
        self.limits = np.array((self.edges[0], self.edges[-1]))
        self.range = self.limits[1] - self.limits[0]

        self.values = values
        if self.values is None:
            self.values = np.zeros(self.shape)

    def copy(self) -> Self:
        return Histogram1D(values=self.values, edges=self.edges)

    def normalize(self) -> None:
        values_sum = np.sum(self.values)
        if values_sum > 0.0:
            self.values = self.values / values_sum / self.cell_volume

    def bin(self, points: np.ndarray, density: bool = True) -> np.ndarray:
        self.values, _ = np.histogram(points, bins=self.edges, density=density)
        return np.copy(self.values)

    def __call__(self, **kwargs) -> np.ndarray:
        return self.bin(**kwargs)


def slice_idx(
    ndim: int,
    axis: int | tuple[int, ...],
    ind: int | tuple[int, ...] | list[tuple[int, ...]],
) -> tuple[slice, ...]:
    """Return planar slice index array.

    Parameters
    ----------
    ndim: int
        The number of elements in the slice index array. (The number of dimensions in the array to be sliced.)
    axis: int or tuple[int, ...]
        The sliced axes.
    ind: int, tuple[int, ...] or list[tuple[int, ...]]
        The indices along the sliced axes. If a tuple is provided, this defines the (min, max) index.

    Returns
    -------
    idx: tuple
        The slice index array. A slice of the array `values` may then be accessed as `values[idx]`.
    """
    # Make list if only one axis provided.
    if type(axis) is int:
        axis = [axis]
        # Can also provide only one axis but provide a tuple (selects range of
        # indices) or list (selects specific indices) for ind.
        if not (type(ind) is int):
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


def slice_(
    hist: Histogram,
    axis: int | tuple[int, ...],
    ind: int | tuple[int, ...] | list[tuple[int, ...]],
    return_indices: bool = False,
) -> Histogram | tuple[Histogram, list[slice]]:

    # Get slice idx
    ndim = hist.ndim
    idx = slice_idx(ndim=hist.ndim, axis=axis, ind=ind)

    # Slice the array
    values_new = hist.values[idx]
    values_new = np.squeeze(values_new)

    # Get remaining array coordinates
    coords_new = []
    for axis in range(ndim):
        if type(idx[axis]) is slice:
            # Selects indices using slice object.
            ilo = idx[axis].start
            if ilo is None:
                ilo = 0

            ihi = idx[axis].stop
            if ihi is None:
                ihi = hist.shape[axis]

            if ihi - ilo > 1:
                coords_new.append(hist.coords[axis][idx[axis]])

        elif (type(idx[axis]) is tuple) and (len(idx[axis]) == 2):
            # Selects indices using a tuple of ints (imin, imax).
            ilo, ihi = idx[axis]

        elif array_like(idx[axis]) and (len(idx[axis]) > 1):
            # Selects indices using a list or ndarray.
            coords_new.append(hist.coords[axis][list(idx[axis])])

        else:
            # Selects a single index; this axis is not in the sliced array.
            continue

    # Return new histogram
    hist_new = Histogram(values=values_new, coords=coords_new)

    if return_indices:
        return hist_new, idx
    return hist_new


def project_values(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    if values.ndim == 1:
        return hist

    # Sum over axes
    if type(axis) is int:
        axis = [axis]
    axis = tuple(axis)
    axis_sum = tuple([i for i in range(values.ndim) if i not in axis])

    values_proj = None
    if len(axis) != values.ndim:
        values_proj = np.sum(values, axis_sum)
    else:
        values_proj = values

    # Order the remaining axes
    loc = list(range(values_proj.ndim))
    destination = np.zeros(values_proj.ndim, dtype=int)
    for i, index in enumerate(np.argsort(axis)):
        destination[index] = i
    for i in range(values_proj.ndim):
        if loc[i] != destination[i]:
            j = loc.index(destination[i])
            values_proj = np.swapaxes(values_proj, i, j)
            (loc[i], loc[j]) = (loc[j], loc[i])

    return values_proj


def project(
    hist: Histogram, axis: int | tuple[int, ...], squeeze: bool = True
) -> Histogram | Histogram1D:

    values_proj = project_values(hist.values, axis=axis)
    if (values_proj.ndim == 1) and squeeze:
        axis = int(np.squeeze(axis))
        coords_proj = hist.coords[axis]
        hist_proj = Histogram1D(values_proj, coords_proj)
    else:
        coords_proj = [hist.coords[i] for i in axis]
        hist_proj = Histogram(values_proj, coords_proj)
    return hist_proj


def project_contour_slice_1d(
    hist: Histogram,
    axis: int = 0,
    lmin: float = 0.0,
    lmax: float = 1.0,
    hist_proj: Histogram = None,
) -> Histogram:
    """Apply contour slice in N- 1 dimensions, then project onto the remaining dimension.

    Parameters
    ----------
    hist: Histogram
        Distribution on N-dimensional hist.
    axis: int
        The projection axis.
    lmin, lmax: float
        Min and max contour levels of the (N-1)-dimensional projection of `values`,
        normalized the range [0, 1].
    hist_proj: Histogram
        The (N-1)-dimensional projection of `hist` onto all dimensions other than `axis`.
        If not provided, it will be computed within the function.

        Shape:
        ```
        tuple([hist.shape[i] for i in range(hist.ndim) if i != axis])
        ```

    Returns
    -------
    hist
        The 1D projection of the (N-1)D slice.
    """
    axis_proj = [i for i in range(hist.ndim) if i != axis]

    if hist_proj is None:
        hist_proj = hist.project(axis_proj)

    hist_proj.values = hist_proj.values / np.max(hist_proj.values)

    idx = slice_idx(
        ndim=hist.ndim,
        axis=axis_proj,
        ind=np.where(
            np.logical_and(
                hist_proj.values >= lmin,
                hist_proj.values <= lmax,
            )
        ),
    )

    # `values[idx]` will give a two-dimensional array. Normally we need to sum over
    # the first axis. If `axis == 0`, we need to sum over the second axis.
    values_proj_1d = np.sum(hist.values[idx], axis=int(axis == 0))
    coords_proj_1d = hist.coords[axis]
    return Histogram1D(values=values_proj_1d, coords=coords_proj_1d)


def project_contour_slice_2d(
    hist: Histogram,
    axis: tuple[int, int] = (0, 1),
    lmin: float = 0.0,
    lmax: float = 1.0,
    hist_proj: Histogram = None,
) -> np.ndarray:
    """Apply contour slice in n - 2 dimensions, then project onto the remaining two dimensions.

    Parameters
    ----------
    hist: Histogram
        Distribution on N-dimensional hist.
    axis: tuple
        The 2D projection axis.
    lmin, lmax: float
        Min and max contour levels of the (n-2)-dimensional projection of `values`,
        normalized the the range [0, 1].
    values_proj: ndarray, shape = [values.shape[i] for i in range(values.ndim) if i != axis]
        The (n-2)-dimensional projection of `values` onto all dimensions other than `axis`.
        (If not provided, it will be computed within the function.)

    Returns
    -------
    hist
        The 2D projection of the (N-2)D slice.
    """
    axis_proj = [k for k in range(hist.ndim) if k not in axis]
    axis_proj = tuple(axis_proj)

    if hist_proj is None:
        hist_proj = project(hist, axis=axis_proj)

    hist_proj.scale_max()

    idx = slice_idx(
        ndim=hist.ndim,
        axis=axis_proj,
        ind=np.where(
            np.logical_and(hist_proj.values >= lmin, hist_proj.values <= lmax)
        ),
    )

    # `values[idx]` will give a three-dimensional array. Normally we need to sum over
    # the first axis. If `axis == (0, 1)`, we need to sum over the third axis.
    # If `axis == (0, n - 1), we need to sum over the second axis.
    _axis_proj = (1, 2)
    if axis == (0, 1):
        _axis_proj = (0, 1)
    elif axis == (0, hist.ndim - 1):
        _axis_proj = (0, 2)

    # Two elements of `idx` will be `slice(None)`; these are the elements in `axis`.
    # These will always be in order. So, if `axis[0] > axis[1]`, we need to flip
    # `axis_proj`. Need a way to handle this automatically.
    if axis[0] > axis[1]:
        _axis_proj = tuple(reversed(_axis_proj))

    values_proj = project_values(hist.values[idx], axis=_axis_proj)
    coords_proj = [hist.coords[_axis] for _axis in _axis_proj]
    return Histogram(values_proj, coords_proj)


def copy_values_into_new_dim(
    values: np.ndarray,
    shape: tuple[int, ...],
    axis: int = -1,
    method: str = "broadcast",
    copy: bool = False,
) -> np.ndarray:
    """Copy hist into one or more new dimensions.

    https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times

    Parameters
    ----------
    values: ndarray
        An n-dimensional hist.
    shape: d-tuple of ints
        The shape of the new dimensions.
    axis: int (0 or -1)
        If 0, the new dimensions will be inserted before the first axis. If -1,
        the new dimensions will be inserted after the last axis. I think
        values other than 0 or -1 should work; this does not currently
        work, at least for `method='broadcast'`, last I checked.
    method: {'repeat', 'broadcast'}
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


def mean(hist: Histogram) -> np.ndarray:
    x = [np.average(C, weights=hist.values) for C in hist.meshgrid()]
    x = np.array(x)
    return x


def cov(hist: Histogram) -> np.ndarray:
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

    ndim = hist.ndim
    if ndim < 3:
        return cov_2d(hist.values, hist.coords)

    S = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i):
            # Compute 2 x 2 covariance matrix from this projection.
            axis = (i, j)
            hist_proj = project(hist, axis)
            S_proj = cov_2d(hist_proj.values, hist_proj.coords)

            # Update elements of n x n covariance matrix. This will update
            # some elements multiple times, but it should not matter.
            S[i, i] = S_proj[0, 0]
            S[j, j] = S_proj[1, 1]
            S[i, j] = S[j, i] = S_proj[0, 1]
    return S


def corr(hist: Histogram) -> np.ndarray:
    return cov_to_corr(cov(hist))


def ellipsoid_radii(
    coords: list[np.ndarray], covariance_matrix: np.ndarray
) -> np.ndarray:
    """Return covariance ellipsoid radii from grid coordinates and covariance matrix.

    Radius is defined as x^T Sigma^-1^T. This function computes the radius
    at every point on the hist.

    This is quite slow when n > 4.

    Parameters
    ----------
    coords: list[np.ndarray]
        Grid coordinates along each dimension.
    covariance_matrix: np.ndarray
        N x N covariance matrix.

    Returns
    -------
    ndarray: np.ndarray
        Radius x^T Sigma^-1^T x at each point in hist.
    """
    shape = tuple([len(c) for c in coords])
    COORDS = np.meshgrid(*coords, indexing="ij")

    S = covariance_matrix
    S_inv = np.linalg.inv(S)

    R = np.zeros(shape)
    for ii in np.ndindex(shape):
        v = np.array([C[ii] for C in COORDS])
        R[ii] = np.sqrt(np.linalg.multi_dot([v.T, S_inv, v]))
    return R


def radial_density(
    hist: Histogram,
    radii: np.ndarray,
    dr: float = None,
) -> np.ndarray:
    """Return average density within ellipsoidal shells.

    Parameters
    ----------
    hist: ndarray
        An n-dimensional hist.
    R: ndarray, same shape as `values`.
        Gives the radius at each point in `values`.
    radii: ndarray, shape (k,)
        Radii at which to evaluate the density.
    dr: float
        The radial shell width.

    Returns
    -------
    ndarray, shape (k,)
        The average density within each ellipsoidal shell.
    """
    values = hist.values
    covariance_matrix = cov(hist)
    R = ellipsoid_radii(hist.coords, covariance_matrix)
    if dr is None:
        dr = 0.5 * np.max(R) / (len(R) - 1)

    values_r = []
    for r in radii:
        values_masked = np.ma.masked_where(np.logical_or(R < r, R > r + dr), values)
        values_r.append(np.mean(values_masked))
    return np.array(values_r)


def blur_hist(hist: Histogram, sigma: float) -> np.ndarray:
    """Call scipy.ndhist.gaussian_filter."""
    hist.values = ndhist.gaussian_filter(hist.values, sigma)
    return hist


def clip_hist(
    hist: Histogram, lmin: float = None, lmax: float = None, frac: bool = False
) -> np.ndarray:
    """Clip between lmin and lmax, can be fractions or absolute values."""
    if not (lmin or lmax):
        return hist
    if frac:
        values = hist.values
        values_max = np.max(values)
        if lmin:
            lmin = values_max * lmin
        if lmax:
            lmax = values_max * lmax
    hist.values = np.clip(values, lmin, lmax)
    return hist


def fill_hist(hist: np.ndarray, fill_value: float = None) -> np.ndarray:
    """Call numpy.ma.filled."""
    hist.values = np.ma.filled(hist.values, fill_value=fill_value)
    return hist


def thresh_hist(hist: Histogram, lmin: float = None, frac: bool = False) -> np.ndarray:
    if lmin:
        if frac:
            lmin = lmin * hist.values.max()
        hist.values[hist.values < lmin] = 0.0
    return hist


def sample_hist(hist: Histogram, size: int = 100, noise: float = 0.0) -> np.ndarray:

    values = hist.values
    edges = hist.edges

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
            points[:, axis] += (
                noise * 0.5 * np.random.uniform(-delta, delta, size=points.shape[0])
            )
    return points


def sample_sparse_hist(
    hist: SparseHistogram, size: int = 100, noise: float = 0.0
) -> np.ndarray:
    """Sample from sparse histogram.

    Parameters
    ----------
    hist: SparseHist
        Sparse N-dimensional hist.
    size: int
        The number of samples to draw.
    noise: float
        Add noise to each particle; a number is drawn uniformly from a box centered on the
        particle with dimensions equal to the histogram bin dimensions. `noise` scales the
        box dimensions relative to the bin dimensions.

    Returns
    -------
    ndarray, shape (size, N)
        Samples drawn from the distribution.
    """
    values = hist.values
    indices = hist.indices
    edges = hist.edges

    shape = [len(e) - 1 for e in edges]
    indices_flat = np.ravel_multi_index(indices.T, shape)
    idx = np.random.choice(
        indices_flat, size=size, replace=True, p=(values / np.sum(values))
    )
    idx = np.unravel_index(idx, shape=shape)
    lb = [edges[axis][idx[axis]] for axis in range(len(shape))]
    ub = [edges[axis][idx[axis] + 1] for axis in range(len(shape))]
    points = np.squeeze(np.random.uniform(lb, ub).T)
    if noise:
        delta = 0.5 * noise * np.array([np.mean(np.diff(e)) for e in edges])
        points += np.random.uniform(low=-delta, high=delta, size=points.shape)
    return points
