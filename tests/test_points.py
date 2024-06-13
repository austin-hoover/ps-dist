import numpy as np
import psdist as ps


def test_sparse_histogram():
    n_points = int(1.00e04)
    n_dims = 4
    n_bins = 10
    rng = np.random.default_rng(1234)

    X = rng.normal(size=(n_points, n_dims))

    edges = ps.points.histogram_bin_edges(X, bins=n_bins)
    (nonzero_indices, nonzero_counts, nonzero_edges) = ps.points.sparse_histogram(X, bins=edges)

    hist, _ = ps.points.histogram(X, bins=edges)
    hist = hist.astype(int)

    assert len(nonzero_counts) == np.count_nonzero(hist)

    for idx, count in zip(nonzero_indices, nonzero_counts):
        assert hist[tuple(idx)] == count
