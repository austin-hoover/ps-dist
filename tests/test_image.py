import numpy as np
import psdist as ps


def test_project():
    values = np.random.normal(size=(6, 4, 12, 8, 2, 9))
    for axis in np.ndindex(*(values.ndim * [values.ndim])):
        if len(np.unique(axis)) != values.ndim:
            continue
        shape = ps.image.project(values, axis).shape
        correct_shape = tuple([values.shape[i] for i in axis])
        assert shape == correct_shape


def test_slice_idx():
    values = np.random.normal(size=(6, 4, 12, 8, 2, 9))
    axis = (2, 0, 3, 5)
    ind = (3, (3, 9), [4, 5, 6], 1)
    idx = ps.image.slice_idx(values.ndim, axis=axis, ind=ind)
    values[idx]
