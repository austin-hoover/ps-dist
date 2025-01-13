import numpy as np
import psdist as ps


def test_project():
    shape = (6, 4, 12, 8, 2, 9)
    values = np.abs(np.random.normal(size=shape))
    coords = [np.arange(s) for s in shape]

    hist = ps.hist.Histogram(values, coords)

    for axis in np.ndindex(*(values.ndim * [values.ndim])):
        if len(np.unique(axis)) != values.ndim:
            continue
        hist_proj = ps.hist.project(hist, axis)
        shape = hist_proj.shape
        correct_shape = tuple([values.shape[i] for i in axis])
        assert shape == correct_shape


def test_slice_idx():
    shape = (6, 4, 12, 8, 2, 9)
    values = np.abs(np.random.normal(size=shape))
    axis = (2, 0, 3, 5)
    ind = (3, (3, 9), [4, 5, 6], 1)
    idx = ps.hist.slice_idx(values.ndim, axis=axis, ind=ind)
    values[idx]
