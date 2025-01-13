import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib import pyplot as plt

from ..cov import rms_ellipse_params as _rms_ellipse_params


def plot_ellipse(
    r1: float = 1.0,
    r2: float = 1.0,
    angle: float = 0.0,
    center: tuple[float, float] = None,
    ax=None,
    **kws,
):
    """Plot ellipse with semi-axes `c1`,`c2` tilted `angle`radians below the x axis."""
    kws.setdefault("fill", False)
    kws.setdefault("color", "black")
    kws.setdefault("lw", 1.25)

    if center is None:
        center = (0.0, 0.0)

    d1 = r1 * 2.0
    d2 = r2 * 2.0
    angle = -np.degrees(angle)
    ax.add_patch(patches.Ellipse(center, d1, d2, angle=angle, **kws))
    return ax


def plot_circle(r: float = 1.0, center: tuple[float, float] = None, ax=None, **kws):
    """Plot circle of radius r."""
    if center is None:
        center = (0.0, 0.0)
    return plot_ellipse(r, r, center=center, ax=ax, **kws)


def plot_rms_ellipse_cov(
    cov_matrix: np.ndarray,
    center: np.ndarray = None,
    level: float = 1.0,
    ax=None,
    **ellipse_kws,
):
    """Plot RMS ellipse from 2 x 2 covariance matrix."""
    if center is None:
        center = (0.0, 0.0)
    if type(level) not in [list, tuple, np.ndarray]:
        level = [level]
    r1, r2, angle = _rms_ellipse_params(cov_matrix)
    for level in level:
        plot_ellipse(
            r1 * level, r2 * level, angle=angle, center=center, ax=ax, **ellipse_kws
        )
    return ax


def cubehelix_cmap(color: str = "red", dark: float = 0.20):
    kws = dict(
        n_colors=12,
        rot=0.0,
        gamma=1.0,
        hue=1.0,
        light=1.0,
        dark=dark,
        as_cmap=True,
    )

    cmap = None
    if color == "red":
        cmap = sns.cubehelix_palette(start=0.9, **kws)
    elif color == "pink":
        cmap = sns.cubehelix_palette(start=0.8, **kws)
    elif color == "blue":
        cmap = sns.cubehelix_palette(start=2.8, **kws)
    else:
        raise ValueError
    return cmap
