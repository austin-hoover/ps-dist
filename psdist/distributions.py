import numpy as np
from psdist.utils import sphere_volume


class Distribution:
    def __init__(self, d=2):
        self.d = d
        
    def prob(self, x):
        return 0.0

    def sample(self, n):
        return

    def entropy(self):
        return


class Gaussian(Distribution):
    def __init__(self, d=2):
        super().__init__(d=d)

    def prob(self, x):
        return np.exp(-0.5 * np.sum(x**2, axis=1)) / np.sqrt(2.0 * np.pi)

    def entropy(self):
        return 0.5 * (np.log(2.0 * np.pi) + 1.0)
 
    def sample(self, n):
        return np.random.normal(size=(n, self.d))


class Waterbag(Distribution):
    def __init__(self, d=2):
        super().__init__(d=d)
        self.r_max = np.sqrt(self.d + 2)

    def prob(self, x):
        p = np.zeros(x.shape[0])
        r = np.sqrt(np.sum(np.square(x), axis=1))
        normalization = sphere_volume(r=self.r_max, d=self.d)
        p[r <= self.r_max] = 1.0 / normalization
        return p
     
    def sample(self, n):
        x = np.random.normal(size=(n, self.d))
        scale = 1.0 / np.sqrt(np.sum(x**2, axis=1))
        x = x * scale[:, None]
        scale = np.random.uniform(0.0, 1.0, size=n) ** (1.0 / self.d)
        scale = scale * self.r_max
        x = x * scale[:, None]
        return x


def gen_dist(name="gaussian", d=2, *args, **kwargs):
    constructors = {
        "gaussian": Gaussian,
        "waterbag": Waterbag,
    }
    constructor = constructors[name]
    return constructor(d=d, *args, **kwargs)





