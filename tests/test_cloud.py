import sys
import numpy as np

sys.path.append("..")
import psdist as ps


def test_sparse_histogram():
    n_points = int(1.00e04)
    n_dims = 4
    n_bins = 10
    state = np.random.RandomState(1234)
    X = state.normal(size=(n_points, n_dims))

    edges = ps.cloud.histogram_bin_edges(X, bins=n_bins)
    nonzero_indices, nonzero_counts, nonzero_edges = ps.cloud.sparse_histogram(
        X, bins=edges
    )
    hist, _ = ps.cloud.histogram(X, bins=edges)
    hist = hist.astype(int)

    assert len(nonzero_counts) == np.count_nonzero(hist)
    for idx, count in zip(nonzero_indices, nonzero_counts):
        assert hist[tuple(idx)] == count
