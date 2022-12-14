"""Input/output."""
import numpy as np


# The following three functions allow saving/loading ragged arrays in .npz format.
# This is useful if we have multiple coordinate arrays with a different
# number of points in each array.
# (Source: https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR)
def stack_ragged(arrays, axis=0):
    """Stacks list of arrays along first axis.

    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(a)[axis] for a in arrays]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(arrays, axis=axis)
    return stacked, idx


def save_stacked_array(filename, arrays, axis=0):
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(arrays, axis=axis)
    np.savez(filename, stacked_array=stacked, stacked_index=idx)


def load_stacked_arrays(filename, axis=0):
    """ "Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(filename)
    idx = npz_file["stacked_index"]
    stacked = npz_file["stacked_array"]
    return np.split(stacked, idx, axis=axis)
