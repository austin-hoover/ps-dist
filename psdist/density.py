from typing import Callable
import numpy as np
import scipy.stats


class DensityEstimator:
    def __init__(self, points: np.ndarray = None) -> None:
        if points is not None:
            self.train(points)

    def train(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def prob(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.prob(points)


class GaussianKDE(DensityEstimator):
    def __init__(
        self, points: np.ndarray = None, bandwidth: float = None, **kws
    ) -> None:
        self.bandwidth = bandwidth
        self.estimator = None
        super().__init__(points, **kws)

    def train(self, points: np.ndarray) -> None:
        self.estimator = scipy.stats.gaussian_kde(points.T)

    def prob(self, points: np.ndarray) -> np.ndarray:
        return self.estimator(points.T)


def estimate_density(
    points: np.ndarray, eval_points: np.ndarray, method: str = "kde", **kws
) -> np.ndarray:
    """Estimate density from samples."""
    estimator = None
    if method == "kde":
        estimator = GaussianKDE(**kws)
    else:
        ValueError(f"Invalid method '{method}'")

    ndim = 1
    if np.ndim(points) > 1:
        ndim = points.shape[1]

    estimator.train(points)
    return estimator.prob(eval_points)
