import numpy as np
import psdist as ps
from psdist import GaussianKDE


def gen_dist(ndim: int) -> np.ndarray:
    return np.random.rand(100, ndim)


def test_sparse_histogram():
    n_points = int(1.00e04)
    n_dims = 4
    n_bins = 10
    rng = np.random.default_rng(1234)

    x = rng.normal(size=(n_points, n_dims))

    edges = ps.histogram_bin_edges(x, bins=n_bins)
    sparse_hist = ps.sparse_histogram(x, bins=edges)

    hist = ps.histogram(x, bins=edges)
    hist.values = hist.values.astype(int)

    nonzero_counts = sparse_hist.values
    nonzero_indices = sparse_hist.indices

    assert len(nonzero_counts) == np.count_nonzero(hist.values)

    for idx, count in zip(nonzero_indices, nonzero_counts):
        assert hist.values[tuple(idx)] == count


def test_get_radii():
    x = gen_dist(ndim=6)
    radii = ps.get_radii(x)
    assert np.all(radii >= 0)
    cov_matrix = np.cov(x.T)
    radii = ps.get_radii(x, cov_matrix)
    assert np.all(radii >= 0)


def test_enclosing_sphere_radius():
    x = gen_dist(ndim=6)
    radius = ps.enclosing_sphere_radius(x)
    assert radius > 0


def test_enclosing_ellipsoid_radius():
    x = gen_dist(ndim=6)
    radius = ps.enclosing_ellipsoid_radius(x)
    assert radius > 0


def test_limits():
    x = gen_dist(ndim=6)
    limits = ps.limits(x)
    limits = ps.limits(x, rms=1.0)
    limits = ps.limits(x, share=(0, 1))
    limits = ps.limits(x, zero_center=True)


def test_project():
    x = gen_dist(ndim=6)
    x_proj = ps.project(x, axis=(0, 1))
    assert np.all(np.equal(x_proj, x[:, (0, 1)]))


def test_transform_linear():
    x = gen_dist(ndim=6)
    matrix = np.random.rand(6, 6)
    y = ps.transform_linear(x, matrix)
    assert y.shape == x.shape


def test_slice():
    x = gen_dist(ndim=6)
    x_slice = ps.slice_(x, axis=(2, 3), center=0.5, width=0.2)
    assert x_slice.shape[1] == 6


def test_slice_sphere():
    x = gen_dist(ndim=6)
    x_slice = ps.slice_sphere(x, rmax=1.0)
    assert np.all(ps.get_radii(x_slice) <= 1.0)


def test_slice_ellipsoid():
    x = gen_dist(ndim=6)
    x_slice = ps.slice_ellipsoid(x, rmax=1.0)


def test_slice_contour():
    x = gen_dist(ndim=2)
    ps.slice_contour(x, lmax=0.5)


def test_normalize_2d_projections():
    x = gen_dist(ndim=6)
    x_new = ps.normalize_2d_projections(x, scale=True)
    assert x.shape == x_new.shape


def test_decorrelate_x_y_z():
    x = gen_dist(ndim=6)
    x_new = ps.decorrelate_x_y_z(x)
    assert x.shape == x_new.shape


def test_downsample():
    x = gen_dist(ndim=6)
    x_new = ps.downsample(x, size=50)
    assert x_new.shape == (50, 6)


def test_histogram_bin_edges():
    x = gen_dist(ndim=2)
    edges = ps.histogram_bin_edges(x, bins=10)
    assert len(edges) == 2


def test_histogram():
    x = gen_dist(ndim=2)
    hist = ps.histogram(x, bins=10)
    assert hist.ndim == 2


def test_sparse_histogram():
    x = gen_dist(ndim=2)
    sparse_hist = ps.sparse_histogram(x, bins=10)
    assert hasattr(sparse_hist, "values")


def test_radial_histogram():
    x = gen_dist(ndim=2)
    hist = ps.radial_histogram(x, bins=10)


def test_density_estimation_gaussian_kde():
    x = gen_dist(ndim=2)
    x_eval = x

    estimator = ps.GaussianKDE(x)
    density = estimator(x_eval)
    assert len(density) == x.shape[0]
