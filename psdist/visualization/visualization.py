import warnings

from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt
import scipy.optimize

import psdist.image
import psdist.cloud
import psdist.utils
import psdist.visualization.cloud as vis_cloud
import psdist.visualization.image as vis_image


def ellipse(c1=1.0, c2=1.0, angle=0.0, center=(0, 0), ax=None, **kws):
    """Plot ellipse with semi-axes `c1`,`c2` tilted `angle`radians below the x axis."""
    kws.setdefault("fill", False)
    kws.setdefault("color", "black")
    width = 2.0 * c1
    height = 2.0 * c2
    return ax.add_patch(
        patches.Ellipse(center, width, height, -np.degrees(angle), **kws)
    )


def circle(r=1.0, center=(0.0, 0.0), ax=None, **kws):
    """Plot a circle."""
    return ellipse(r, r, center=center, ax=ax, **kws)


def rms_ellipse_dims(Sigma, axis=(0, 1)):
    """Return dimensions of projected rms ellipse.

    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        The phase space covariance matrix.
    axis : 2-tuple
        The axis on which to project the covariance ellipsoid. Example: if the
        axes are {x, xp, y, yp}, and axis=(0, 2), then the four-dimensional
        ellipsoid is projected onto the x-y plane.
    ax : plt.Axes
        The ax on which to plot.

    Returns
    -------
    c1, c2 : float
        The ellipse semi-axis widths.
    angle : float
        The tilt angle below the x axis [radians].
    """
    i, j = axis
    sii, sjj, sij = Sigma[i, i], Sigma[j, j], Sigma[i, j]
    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)
    sin, cos = np.sin(angle), np.cos(angle)
    sin2, cos2 = sin**2, cos**2
    c1 = np.sqrt(abs(sii * cos2 + sjj * sin2 - 2 * sij * sin * cos))
    c2 = np.sqrt(abs(sii * sin2 + sjj * cos2 + 2 * sij * sin * cos))
    return c1, c2, angle


def rms_ellipse(Sigma=None, center=None, level=1.0, ax=None, **ellipse_kws):
    """Plot RMS ellipse from 2 x 2 covariance matrix."""
    if type(level) not in [list, tuple, np.ndarray]:
        level = [level]
    c1, c2, angle = rms_ellipse_dims(Sigma)
    for level in level:
        _c1 = c1 * level
        _c2 = c2 * level
        ellipse(_c1, _c2, angle=angle, center=center, ax=ax, **ellipse_kws)
    return ax


def fit_linear(x, y):
    """Return (yfit, slope, intercept) from linear fit."""

    def func(x, slope, intercept):
        return slope * x + intercept

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    slope, intercept = popt
    return func(x, slope, intercept), slope, intercept


def fit_normal(x, y):
    """Return (yfit, sigma, mu, amplitude, offset) from Gaussian fit."""

    def func(x, sigma, mu, amplitude, offset):
        amplitude = amplitude / (sigma * np.sqrt(2.0 * np.pi))
        return offset + amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    sigma, mu, amplitude, offset = popt
    return func(x, sigma, mu, amplitude, offset), sigma, mu, amplitude, offset


def scale_profile(profile, scale=None, edges=None, coords=None):
    if not scale:
        return profile
    if np.max(profile) <= 0.0:
        return profile
    if scale == "density":
        if edges is None:
            if coords is not None:
                edges = psdist.utils.edges_from_centers(coords)
            else:
                edges = np.arange(len(profile) + 1)
        return profile / np.sum(profile * np.diff(edges))
    elif scale == "max":
        return profile / np.max(profile)
    elif type(scale) in [int, float]:
        return profile / scale
    else:
        return profile


def plot_profile(
    profile,
    coords=None,
    edges=None,
    ax=None,
    orientation="vertical",
    kind="line",
    fill=False,
    offset=0.0,
    scale=None,
    **kws,
):
    """Plot one-dimensional profile.

    Parameters
    ----------
    profile : ndarray, shape (n,)
        A one-dimensional profile.
    coords : ndarray, shape (n,)
        Coordinate of the points in `profile`. Can be None if `edges` is not None.
    edges : ndarray, shape (n + 1,)
        Bin edges (if the profile is a histogram). Can be None if `centers` is not None.
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

    if coords is None and edges is None:
        raise ValueError("coords or edges must be provided")
    if coords is None and edges is not None:
        coords = psdist.utils.centers_from_edges(edges)
    if edges is None and coords is not None:
        edges = psdist.utils.edges_from_centers(coords)

    coords = np.array(coords)
    edges = np.array(edges)
    profile = np.array(profile)
    profile = scale_profile(profile, scale=scale, edges=edges)

    if kind == "step":
        return ax.stairs(
            profile + offset,
            edges=edges,
            fill=fill,
            baseline=offset,
            orientation=orientation,
            **kws,
        )
    if kind == "line":
        profile = profile + offset
        if fill:
            if orientation == "horizontal":
                return ax.fill_betweenx(coords, offset, profile, **kws)
            else:
                return ax.fill_between(coords, offset, profile, **kws)
        else:
            coords = np.hstack([coords[0], coords, coords[-1]])
            profile = np.hstack([offset, profile, offset])
            if orientation == "horizontal":
                return ax.plotx(coords, profile, **kws)
            else:
                return ax.plot(coords, profile, **kws)
    elif kind == "bar":
        if orientation == "horizontal":
            return ax.barh(coords, profile, left=(offset * np.ones(len(coords))), **kws)
        else:
            return ax.bar(
                coords, profile, bottom=(offset * np.ones(len(coords))), **kws
            )
    else:
        raise ValueError("Invalid plot kind")


def combine_limits(limits_list):
    """Combine a stack of limits, keeping min/max values.

    Example: [[(-1, 1), (-3, 2)], [(-1, 2), (-1, 3)]] --> [(-1, 2), (-3, 3)].

    Parameters
    ----------
    limits_list : list[list[tuple]]
        Each element is a set of limits [(xmin, xmax), (ymin, ymax), ...].

    Returns
    -------
    list[tuple]
        New set of limits [(xmin, xmax), (ymin, ymax), ...].
    """
    limits_list = np.array(limits_list)
    mins = np.min(limits_list[:, :, 0], axis=0)
    maxs = np.max(limits_list[:, :, 1], axis=0)
    return list(zip(mins, maxs))


def center_limits(limits):
    """Center limits at zero.

    Example: [(-3, 1), (-4, 5)] --> [(-3, 3), (-5, 5)].

    Parameters
    ----------
    limits : list[tuple]
        A set of limits [(xmin, xmax), (ymin, ymax), ...].

    Returns
    -------
    limits : list[tuple]
        A new set of limits centered at zero [(-x, x), (-y, y), ...].
    """
    mins, maxs = list(zip(*limits))
    maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
    return list(zip(-maxs, maxs))
