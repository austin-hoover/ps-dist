from matplotlib import patches
import numpy as np
import scipy.optimize
import ultraplot as uplt

import psdist as ps
from psdist.hist import Histogram
from psdist.hist import Histogram1D

from .utils import scale_profile

# from psdist.cov import rms_ellipse_dims
# from psdist.plot.hist import plot as plot_image
# from psdist.plot.points import plot as plot_points


def plot_ellipse(
    r1: float = 1.0,
    r2: float = 1.0,
    angle: float = 0.0,
    center: tuple[float, float] = None,
    ax=None,
    **kws,
) -> uplt.Axes:
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


def plot_circle(r: float = 1.0, center: tuple[float, float] = None, ax=None, **kws) -> uplt.Axes:
    """Plot circle of radius r."""
    if center is None:
        center = (0.0, 0.0)
    return plot_ellipse(r, r, center=center, ax=ax, **kws)


def plot_rms_ellipse(
    cov_matrix: np.ndarray, center: np.ndarray = None, level: float = 1.0, ax=None, **ellipse_kws
) -> uplt.Axes:
    """Plot RMS ellipse from 2 x 2 covariance matrix."""
    if center is None:
        center = (0.0, 0.0)
    if type(level) not in [list, tuple, np.ndarray]:
        level = [level]
    r1, r2, angle = ps.cov.rms_ellipse_params(cov_matrix)
    for level in level:
        plot_ellipse(r1 * level, r2 * level, angle=angle, center=center, ax=ax, **ellipse_kws)
    return ax


def plot_profile(
    profile: Histogram1D,
    ax=None,
    orientation: str = "vertical",
    kind: str = "line",
    fill: bool = False,
    offset: float = 0.0,
    scale: float = None,
    **kws,
):
    """Plot one-dimensional profile.

    Parameters
    ----------
    profile : Histogram1d
        A one-dimensional distribution.
    ax : Axes
        The plotting axis.
    orientation : {"vertical", "horizontal"}
        Whether to plot on the x or y axis.
    kind : {"line", "step", "bar"}
        Whether to plot a line or a piecewise-constant curve.
        "line" calls `ax.plot`, `ax.plotx`, `ax.fill_between`, or `ax.fill_between_x`.
        "step" calls `ax.stairs`.
        "bar" calls `ax.bar` or `ax.barh`.
    fill : bool
        Whether to fill below the curve.
    offset : float
        Offset applied to the profile.
    scale : {None, "density", "max", float}
        Scale the profile by density (area under curve), max value, or value provided.
    **kws
        Key word arguments passed to the plotting function.
    """
    kws.setdefault("lw", 1.5)

    profile = scale_profile(profile, scale=scale)
    values = profile.values.copy()
    coords = profile.coords.copy()
    edges = profile.edges.copy()

    if kind == "step":
        return ax.stairs(
            values + offset,
            edges=edges,
            fill=fill,
            baseline=offset,
            orientation=orientation,
            **kws,
        )
    if kind == "line":
        values += offset
        if fill:
            if orientation == "horizontal":
                return ax.fill_betweenx(coords, offset, values, **kws)
            else:
                return ax.fill_between(coords, offset, values, **kws)
        else:
            coords = np.hstack([coords[0], coords, coords[-1]])
            values = np.hstack([offset, values, offset])
            if orientation == "horizontal":
                return ax.plotx(coords, values, **kws)
            else:
                return ax.plot(coords, values, **kws)
    elif kind == "bar":
        pad = offset * np.ones(len(coords))
        if orientation == "horizontal":
            return ax.barh(coords, values, left=pad, **kws)
        else:
            return ax.bar(
                coords, values, bottom=pad, **kws
            )
    else:
        raise ValueError(f"Invalid plot kind '{kind}'")


# def combine_limits(limits_list):
#     """Combine a stack of limits, keeping min/max values.
#
#     Example: [[(-1, 1), (-3, 2)], [(-1, 2), (-1, 3)]] --> [(-1, 2), (-3, 3)].
#
#     Parameters
#     ----------
#     limits_list : list[list[tuple]]
#         Each element is a set of limits [(xmin, xmax), (ymin, ymax), ...].
#
#     Returns
#     -------
#     list[tuple]
#         New set of limits [(xmin, xmax), (ymin, ymax), ...].
#     """
#     limits_list = np.array(limits_list)
#     mins = np.min(limits_list[:, :, 0], axis=0)
#     maxs = np.max(limits_list[:, :, 1], axis=0)
#     return list(zip(mins, maxs))
#
#
# def center_limits(limits):
#     """Center limits at zero.
#
#     Example: [(-3, 1), (-4, 5)] --> [(-3, 3), (-5, 5)].
#
#     Parameters
#     ----------
#     limits : list[tuple]
#         A set of limits [(xmin, xmax), (ymin, ymax), ...].
#
#     Returns
#     -------
#     limits : list[tuple]
#         A new set of limits centered at zero [(-x, x), (-y, y), ...].
#     """
#     mins, maxs = list(zip(*limits))
#     maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
#     return list(zip(-maxs, maxs))
#
#
# def cubehelix_cmap(color: str = "red", dark: float = 0.20):
#     import seaborn as sns
#
#     kws = dict(
#         n_colors=12,
#         rot=0.0,
#         gamma=1.0,
#         hue=1.0,
#         light=1.0,
#         dark=dark,
#         as_cmap=True,
#     )
#
#     cmap = None
#     if color == "red":
#         cmap = sns.cubehelix_palette(start=0.9, **kws)
#     elif color == "pink":
#         cmap = sns.cubehelix_palette(start=0.8, **kws)
#     elif color == "blue":
#         cmap = sns.cubehelix_palette(start=2.8, **kws)
#     else:
#         raise ValueError
#     return cmap
