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


def plot1d(x, y, ax=None, offset=0.0, flipxy=False, kind="line", **kws):
    """Convenience function for one-dimensional line/step/bar plots."""
    kws.setdefault("color", "black")

    func = ax.plot
    if kind in ["line", "step"]:
        if flipxy:
            func = ax.plotx
        else:
            func = ax.plot
        if kind == "step":
            kws.setdefault("drawstyle", "steps-mid")
    elif kind in ["linefilled", "stepfilled"]:
        if flipxy:
            func = ax.fill_betweenx
        else:
            func = ax.fill_between
        kws.setdefault("alpha", 1.0)
        if kind == "stepfilled":
            kws.setdefault("step", "mid")
    elif kind == "bar":
        if flipxy:
            func = ax.barh
        else:
            func = ax.bar

    # Handle offset
    if kind == "bar":
        kws["left" if flipxy else "bottom"] = offset * np.ones(len(x))
        return func(x, y, **kws)
    elif kind in ["linefilled", "stepfilled"]:
        return func(x, offset, y + offset, **kws)
    return func(x, y + offset, **kws)


def stack_limits(limits_list):
    limits_list = np.array(limits_list)
    mins = np.min(limits_list[:, :, 0], axis=0)
    maxs = np.max(limits_list[:, :, 1], axis=0)
    return [(mins[i], maxs[i]) for i in range(len(mins))]
