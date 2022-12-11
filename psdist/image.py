"""Functions for n-dimensional images."""
import numpy as np
from tqdm import trange
from tqdm import tqdm
from scipy import ndimage

from . import utils


def get_grid_coords(*coords):
    """Return list of grid coordinates from coordinate arrays along each axis.
    
    Parameters
    ----------
    coords : list of 1D arrays
        Coordinates of each axis of regular grid. Example: [[1, 2, 3], [0, 1, 2]].
    
    Returns
    -------
    ndarray, shape (K, len(coords))
        Coordinate array for all points in the grid. The total number of grid 
        points is `K = np.prod([len(c) for c in coords])`.
    """
    return np.vstack([C.ravel() for C in np.meshgrid(*coords, indexing='ij')]).T


def max_indices(f):
    """Return the indices of the maximum element of `f`."""
    return np.unravel_index(np.argmax(f), f.shape) 


def make_slice(n, axis=0, ind=0):
    """Return planar slice index array.
    
    Parameters
    ----------
    n : int
        The number of elements in the slice index array. (The number of dimensions
        in the array to be sliced.)
    axis : list[int]
        The sliced axes.
    ind : list[int] or list[tuple]
        The indices along the sliced axes. If a tuple is provided, this
        defines the (min, max) index.
        
    Returns
    -------
    idx : n-tuple
        The slice index array. A slice of the array `f` may then be accessed as
        `f[idx]`.
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
    idx = n * [slice(None)]
    # If any indices were provided, add them to `idx`.
    for k, i in zip(axis, ind):
        if i is None:
            continue
        elif type(i) is tuple and len(i) == 2:
            idx[k] = slice(i[0], i[1])
        else:
            idx[k] = i
    return tuple(idx)
    
    
def make_slice_ellipsoid(f, axis, rmin=0.0, rmax=1.0):
    """Compute an ellipsoid slice.
    
    Ellipsoid is computed from the covariance matrix of `f`. 
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    axis : list[int]
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
        A version of `f` in which elements outside the slice are masked.
    """
    raise NotImplementedError
    
    
def make_slice_contour(f, axis, lmin=0.0, lmax=1.0):
    """Compute a contour slice.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    axis : list[int]
        Specificies the subspace in which the contours are computed. (See
        `make_slice_ellipsoid`.)
    lmin, lmax : float
        `f`is projected onto `axis` and the projection `fpr` is normalized to 
        the range [0, 1]. Then, we find the points in this subspace such that 
        `fpr` is within the range [lmin, lmax]. 
    
    Returns
    -------
    np.ma.masked_array
        A version of `f` in which elements outside the slice are masked.
    """
    raise NotImplementedError


def project(f, axis=0):
    """Project along one or more axes.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    axis : list[int]
        The axes onto which the image is projected, i.e., the
        axes which are not summed over. Can be an int or list
        of ints. Array axes are swapped as required.
    
    Returns
    -------
    proj : ndarray
        The projection of `image` onto the specified axis.
    """
    # Sum over specified axes.
    n = f.ndim
    if type(axis) is int:
        axis = [axis]
    axis = tuple(axis)
    axis_sum = tuple([i for i in range(f.ndim) if i not in axis])
    proj = np.sum(f, axis_sum)
    
    # Order the remaining axes.
    n = proj.ndim
    loc = list(range(n))
    destination = np.zeros(n, dtype=int)
    for i, index in enumerate(np.argsort(axis)):
        destination[index] = i
    for i in range(n):
        if loc[i] != destination[i]:
            j = loc.index(destination[i])
            proj = np.swapaxes(proj, i, j)
            loc[i], loc[j] = loc[j], loc[i]
    return proj


def project1d_contour(f, axis=0, lmin=0.0, lmax=1.0, fpr=None, normalize=True, return_frac=False):
    """Apply contour slice in n-1 dimensions, then project onto the other dimension.
        
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    axis : int
        The projection axis.
    lmin, lmax : float
        See `make_slice_contour`.
    fpr : ndarray, shape [f.shape[i] for i in range(f.ndim) if i != axis]
        The (n-1)-dimensional projection of `f` onto all dimensions other than `axis`. This
        is an optional parameter; if not provided, it will be computed within the function.
    normalize : bool
        Whether to normalize the projection (so that its sum is unity).
    return_frac : bool
        Whether to return the fractional density of `f` selected by the slice.
    
    Returns
    -------
    p : ndarray, shape f.shape[axis]
        The 1D projection within the specified boundary.
    """
    axis_proj = [i for i in range(f.ndim) if i != axis]
    if fpr is None:
        fpr = project(f, axis_proj)
    fpr = fpr / np.max(fpr)        
    ind = np.where(np.logical_and(fpr >= lmin, fpr <= lmax))
    frac = np.sum(fpr[ind]) / np.sum(fpr)
    idx = make_slice(f.ndim, axis_proj, ind)    
    p = np.sum(f[idx], axis=int(axis == 0))
    if normalize:
        p = p / np.sum(p)
    if return_frac:
        return p, frac
    return p


def project2d_contour(f, axis=(0, 1), lmin=0.0, lmax=1.0, fpr=None, normalize=True, return_frac=False):
    """Apply contour slice in n-2 dimensions, then project onto the other two dimensions.
    
    The parameters are defined in `project1d_contour`.
    """
    # Compute a 3D mask.
    axis_proj = [i for i in range(f.ndim) if i not in axis]
    if fpr is None:
        fpr = project(f, axis_proj)
    fpr = fpr / np.max(fpr)
    mask = np.logical_or(fpr < lmin, fpr > lmax)
    frac = np.sum(fpr[~mask]) / np.sum(fpr)

    # Copy the 3D mask into the two projected dimensions.
    mask = copy_into_new_dim(mask, (f.shape[axis[0]], f.shape[axis[1]]), axis=-1, copy=True)
    # Put the dimensions in the correct order.
    isort = np.argsort(list(axis_proj) + list(axis))
    mask = np.moveaxis(mask, isort, np.arange(5))
    # Project the masked `f` onto the specified axis.    
    proj = project(np.ma.masked_array(f, mask=mask), axis=axis)
    
    if normalize:
        proj = proj / np.sum(proj)
    if return_frac:
        return proj, frac
    return proj


def copy_into_new_dim(f, shape, axis=-1, method='broadcast', copy=False):
    """Copy image into one or more new dimensions.
    
    See 'https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times'
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    shape : n-tuple of ints
        The shape of the new dimensions.
    axis : int (0 or -1)
        If 0, the new dimensions will be inserted before the first axis. If -1, 
        the new dimensions will be inserted after the last axis. I think
        values other than 0 or -1 should work; this does not currently 
        work for `method='broadcast'`.
    method : {'repeat', 'broadcast'}
        Whether to use `np.repeat` or `np.expand_dims` and `np.broadcast_to`. The
        'broadcast' method is faster.
    """
    if type(shape) in [int, np.int32, np.int64]:
        shape = (shape,)
    if method == 'repeat':
        for i in range(len(shape)):
            f = np.repeat(np.expand_dims(f, axis), shape[i], axis=axis)
        return f
    elif method == 'broadcast':
        if axis == 0:
            new_shape = shape + f.shape
        elif axis == -1:
            new_shape = f.shape + shape
        else:
            raise ValueError('Cannot yet handle axis != 0, -1.')
        for _ in range(len(shape)):
            f = np.expand_dims(f, axis)
        if copy:
            return np.broadcast_to(f, new_shape).copy()
        else:
            return np.broadcast_to(f, new_shape)
    return None


def get_radii(coords, Sigma):
    """Return "radii" (x^T Sigma^-1^T x) from grid coordinates and covariance matrix.
    
    Parameters
    ----------
    coords : list[ndarray], length n
        Coordinate array for each dimension of the regular grid.
    Sigma : ndarray, shape (n, n)
        Covariance matrix of some distribution on the grid.
    
    Returns
    -------
    R : ndarray
        "Radius" x^T Sigma^-1^T x at each point in grid.
    """
    COORDS = np.meshgrid(*coords, indexing='ij')
    shape = tuple([len(c) for c in coords])
    R = np.zeros(shape)
    Sigma_inv = np.linalg.inv(Sigma)
    for ii in tqdm(np.ndindex(shape)):
        vec = np.array([C[ii] for C in COORDS])
        R[ii] = np.sqrt(np.linalg.multi_dot([vec.T, Sigma_inv, vec]))
    return R


def radial_density(f, R, radii, dr=None):
    """Return average density within ellipsoidal shells.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    R : ndarray, same shape as `f`.
        Gives the "radius" at each pixel in f.
    radii : ndarray, shape (k,)
        Radii at which to evaluate the density.
    dr : float
        The shell width.
        
    Returns
    -------
    fr : ndarray, shape (k,)
        The average density within each ellipsoidal shell.
    """
    if dr is None:
        dr = 0.5 * np.max(R) / (len(R) - 1)
    fr = []
    for r in tqdm(radii):
        f_masked = np.ma.masked_where(np.logical_or(R < r, R > r + dr), f)
        # mean density within this shell...
        fr.append(np.mean(f_masked))
    return np.array(fr)


def cov(f, coords, disp=False):
    """Compute the NxN covariance matrix.
    
    To-do: rewrite. The second-order moments can be computed from the
    N(N-1)/2 two-dimensional projections of the image. We do not
    need to loop over every pixel.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
    coords : list[ndarray]
        List of coordinates along each axis of `H`. Can also
        provide meshgrid coordinates.
        
    Returns
    -------
    Sigma : ndarray, shape (n, n)
        The covariance matrix.
    means : ndarray, shape (n,)
        The centroid coordinates.
    """
    if disp:
        print(f'Forming {f.shape} meshgrid...')
    if coords[0].ndim == 1:
        COORDS = np.meshgrid(*coords, indexing='ij')
    n = f.ndim
    f_sum = np.sum(f)
    if f_sum == 0:
        return np.zeros((n, n)), np.zeros((n,))
    if disp:
        print('Averaging...')
    means = np.array([np.average(C, weights=f) for C in COORDS])
    Sigma = np.zeros((n, n))
    _range = trange if disp else range
    for i in _range(Sigma.shape[0]):
        for j in _range(i + 1):
            X = COORDS[i] - means[i]
            Y = COORDS[j] - means[j]
            EX = np.sum(X * f) / f_sum
            EY = np.sum(Y * f) / f_sum
            EXY = np.sum(X * Y * f) / f_sum
            Sigma[i, j] = EXY - EX * EY
    Sigma = utils.symmetrize(Sigma)
    if disp:
        print('Done.')
    return Sigma, means
