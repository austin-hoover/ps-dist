import sys

import numpy as np

sys.path.append("..")
import psdist as ps


n_points = int(1.00e+04)
n_dims = 6
n_bins = 10

state = np.random.RandomState(1234)
X = state.normal(size=(n_points, n_dims))

edges = ps.cloud.histogram_bin_edges(X, bins=n_bins)

nonzero_indices, nonzero_counts, nonzero_edges = ps.cloud.sparse_histogram(X, bins=edges)
print(f"Nonzero bins from sparse hist: {len(nonzero_counts)}")

hist, _ = ps.cloud.histogram(X, bins=edges)
print(f"Nonzero bins from hist: {hist[hist > 0].size}")
print(f"Total bins: {hist.size}")
print(f"Frac nonzero: {hist[hist > 0].size / hist.size}")