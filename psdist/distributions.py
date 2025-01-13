import numpy as np

from .utils import sphere_volume


class Distribution:
    def __init__(self, ndim: int) -> None:
        self.ndim = ndim

    def prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, size: int) -> np.ndarray:
        raise NotImplementedError


class Gaussian(Distribution):
    def __init__(self, ndim: int) -> None:
        super().__init__(ndim=ndim)

    def prob(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * np.sum(x**2, axis=1)) / np.sqrt(2.0 * np.pi)

    def entropy(self):
        return 0.5 * (np.log(2.0 * np.pi) + 1.0)

    def sample(self, size: int) -> np.ndarray:
        return np.random.normal(size=(size, self.ndim))


class Waterbag(Distribution):
    def __init__(self, ndim: int) -> None:
        super().__init__(ndim=ndim)
        self.r_max = np.sqrt(self.ndim + 2)

    def prob(self, x: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(np.square(x), axis=1))
        prob = np.zeros(x.shape[0])
        prob[r <= self.r_max] = 1.0 / sphere_volume(self.r_max, self.ndim)
        return prob

    def sample(self, size: int) -> np.ndarray:
        x = np.random.normal(size=(size, self.ndim))
        scale = 1.0 / np.sqrt(np.sum(x**2, axis=1))
        x = x * scale[:, None]
        scale = np.random.uniform(0.0, 1.0, size=size) ** (1.0 / self.ndim)
        scale = scale * self.r_max
        x = x * scale[:, None]
        return x


def get_distribution(name: str, ndim: int, *args, **kwargs) -> Distribution:
    constructors = {
        "gaussian": Gaussian,
        "waterbag": Waterbag,
    }
    constructor = constructors[name]
    return constructor(ndim=ndim, *args, **kwargs)
