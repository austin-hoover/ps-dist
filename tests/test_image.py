import sys
import numpy as np
import psdist as ps


def test_project():
    f = np.random.normal(size=(6, 4, 12, 8, 2, 9))
    for axis in np.ndindex(*(f.ndim * [f.ndim])):
        if len(np.unique(axis)) != f.ndim:
            continue
        shape = ps.image.project(f, axis).shape
        correct_shape = tuple([f.shape[k] for k in axis])
        assert shape == correct_shape


def test_slice_idx():
    f = np.random.normal(size=(6, 4, 12, 8, 2, 9))
    axis = (2, 0, 3, 5)
    ind = (3, (3, 9), [4, 5, 6], 1)
    idx = ps.image.slice_idx(f.ndim, axis=axis, ind=ind)
    f[idx]