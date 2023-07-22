import sys

import numpy as np

sys.path.append("..")
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
    return
