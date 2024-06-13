from typing import Any
from typing import Union

import numpy as np
import scipy.special


def array_like(array: Any) -> None:
    """Return true if array like."""
    return np.ndim(np.array(array, dtype=object)) > 0


def symmetrize(a: np.ndarray) -> np.ndarray:
    """Return a symmetrized version of array.

    array : A square upper or lower triangular matrix.
    """
    return array + array.T - np.diag(array.diagonal())


def random_choice_no_replacement(array: np.ndarray, size: int) -> np.ndarray:
    """Choose random elements from array without replacement.

    If 0 < size < 1, we select `size * len(array)` elements.
    """
    if type(array) in (list, tuple):
        array = np.array(array)

    if size is None:
        return array

    if size < 0 or size > array.shape[0]:
        raise ValueError("Number of samples must be < number of points.")
    if 0 < size < 1:
        size = size * array.shape[0]
    size = int(size)

    idx = np.random.choice(array.shape[0], size, replace=False)
    return array[idx]


def sphere_surface_area(r: float = 1.0, ndim: int = 3) -> float:
    factor = 2.0 * np.pi ** (0.5 * ndim)
    factor = factor / scipy.special.gamma(0.5 * ndim)
    return factor * (r ** (ndim - 1))


def sphere_volume(r: float = 1.0, ndim: int = 3) -> float:
    factor = (np.pi ** (0.5 * ndim)) / scipy.special.gamma(1.0 + 0.5 * ndim)
    return factor * (r**ndim)


def sphere_shell_volume(rmin: float = 0.0, rmax: float = 1.0, ndim: int = 3):
    return sphere_volume(rmax, ndim=ndim) - sphere_volume(rmin, ndim)


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def coords_from_edges(edges: np.ndarray) -> np.ndarray:
    """Compute bin center coordinates from evenly spaced bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def edges_from_coords(coords: np.ndarray) -> np.ndarray:
    """Compute bin edges from evenly spaced bin center coordinates."""
    delta = np.diff(coords)[0]
    return np.hstack([coords - 0.5 * delta, [coords[-1] + 0.5 * delta]])


def coords_list_from_edges_list(edges_list: list[np.ndarray]) -> list[np.ndarray]:
    return [coords_from_edges(edges) for edges in edges_list]


def edges_list_from_coords_list(coords_list: list[np.ndarray]) -> list[np.ndarray]:
    return [edges_from_coords(coords) for coords in coords_list]


# The following three functions allow saving/loading ragged arrays in .npz format.
# This is useful if we have multiple coordinate arrays with a different number of
# points in each array.
# (Source: https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR)
def stack_ragged(arrays: list[np.ndarray], axis: int = 0) -> tuple[np.ndarray]:
    """Stacks list of arrays along first axis.

    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(a)[axis] for a in arrays]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(arrays, axis=axis)
    return stacked, idx


def save_stacked_array(path: str, arrays: list[np.ndarray], axis: int = 0) -> None:
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(arrays, axis=axis)
    np.savez(path, stacked_array=stacked, stacked_index=idx)


def load_stacked_arrays(path: str, axis: int = 0) -> list[np.ndarray]:
    """ "Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(path)
    idx = npz_file["stacked_index"]
    stacked = npz_file["stacked_array"]
    return np.split(stacked, idx, axis=axis)
