"""Plotting routines for phase space distributions."""
from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt
import scipy.optimize
import scipy.stats

from .. import bunch as psb
from .. import image as psi
from .. import utils
from ..utils import centers_from_edges
from ..utils import edges_from_centers

from . import bunch as _vis_bunch
from . import image as _vis_image


# General
# ------------------------------------------------------------------------------
def ellipse(c1=1.0, c2=1.0, angle=0.0, center=(0, 0), ax=None, **kws):
    """Plot ellipse with semi-axes `c1`,`c2` tilted `angle`radians below the x axis."""
    kws.setdefault("fill", False)
    kws.setdefault("color", "black")
    width = 2.0 * c1
    height = 2.0 * c2
    return ax.add_patch(
        patches.Ellipse(center, width, height, -np.degrees(angle), **kws)
    )


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


def linear_fit(x, y):
    """Return (yfit, slope, intercept) from linear fit."""

    def fit(x, slope, intercept):
        return slope * x + intercept

    popt, pcov = scipy.optimize.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def plot1d(x, y, ax=None, offset=0.0, flipxy=False, kind='line', **kws):
    """Convenience function for one-dimensional line/step/bar plots."""  
    y = y.copy()
    func = ax.plot
    if kind in ['line', 'step']:
        if flipxy:
            func = ax.plotx
        else:
            func = ax.plot
        if kind == 'step':
            kws.setdefault('drawstyle', 'steps-mid')
    elif kind in ['linefilled', 'stepfilled']:
        if flipxy:
            func = ax.fill_betweenx
        else:
            func = ax.fill_between
        kws.setdefault('alpha', 1.0)
        if kind == 'stepfilled':
            kws.setdefault('step', 'mid')
    elif kind == 'bar':
        if flipxy:
            func = ax.barh
        else:
            func = ax.bar
            
    # Handle offset
    if kind == 'bar':
        kws['left' if flipxy else 'bottom'] = offset
        return func(x, y, **kws)
    elif kind in ['linefilled', 'stepfilled']:
        return func(x, offset, y + offset, **kws)
    return func(x, y + offset, **kws)


# Image
# ------------------------------------------------------------------------------
# Corner plot

def _set_corner_limits(axes, limits, diag=False):
    for i in range(axes.shape[1]):
        axes[:, i].format(xlim=limits[i])
    start = int(diag)
    for i, lim in enumerate(limits[1:], start=start):
        axes[i, :(i + 1 - start)].format(ylim=lim)
    return axes
    

def _setup_corner(n, diag, labels, limits=None, **fig_kws):
    """Set up corner plot axes."""
    if labels is None:
        labels = n * [""]
    nrows = ncols = n if diag else n - 1
    start = 1 if diag else 0
    fig_kws.setdefault("figwidth", 1.5 * nrows)
    fig_kws.setdefault("aligny", True)
    fig, axes = pplt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=1,
        sharey=1,
        spanx=False,
        spany=False,
        **fig_kws,
    )
    for i in range(nrows):
        for j in range(ncols):
            if j > i:
                axes[i, j].axis("off")
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[(1 if diag else 0) :, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(nrows):
        axes[:-1, i].format(xticklabels=[])
        axes[i, 1:].format(yticklabels=[])
    for ax in axes:
        ax.format(xspineloc="bottom", yspineloc="left")
    if diag:
        for i in range(n):
            axes[i, i].format(yspineloc="neither")
    if limits is not None:
        _set_corner_limits(axes, limits, diag=diag)
    axes.format(
        xtickminor=True, ytickminor=True, xlocator=("maxn", 3), ylocator=("maxn", 3)
    )
    return fig, axes


def corner(
    data,
    kind="hist",
    diag_kind="step",
    coords=None,
    limits=None,
    labels=None,
    samples=None,
    diag_height_frac=0.6,
    autolim_kws=None,
    diag_kws=None,
    fig_kws=None,
    prof=False,
    prof_kws=None,
    return_fig=False,
    return_mesh=False,
    axes=None,
    **plot_kws,
):
    """Plot one- and two-dimensional projections in a corner plot.

    Parameters
    ----------
    data : ndarray
        If `data.ndim == 2`, we have the coordinates of k points in n-dimensional
        space; otherwise, we have an n-dimensional image.
    kind : {'hist', 'scatter'}
        The kind of 2D plot to make if `data` is a list of points.
            'hist': calls `hist`
            'scatter': `scatter`
            'kde': calls `kde`
    diag_kind : {'line', 'step', 'bar', 'None'}
        Kind of 1D plot on diagonal axes. Any variant of 'None', 'none', None
        will remove the diagonal axes from the figure, resulting in a D-1 x D-1
        array of subplots instead of a D x D array.
    coords : list[ndarray]
        Coordinates along each axis of the grid (if `data` is an image).
    limits : list[tuple]
        The (min, max) coordinates for each dimension. This is used to set the
        axis limits, as well as for data binning if plotting a histogram.
    labels : list[str]
        The axis labels.
    samples : int or float
        Number of samples to use in bivariate plots. If less than 1, specifies
        the fraction of points. All one-dimensional histograms are computed
        before downsampling.
    diag_height_frac : float
        Reduce the height of 1D profiles (diagonal subplots) relative to the
        y axis height.
    autolim_kws : dict
        Key word arguments passed to `autolim`.
    diag_kws : dict
        Key word arguments passed to 1D plotting function on diagonal axes.
    fig_kws : dict
        Key word arguments passed to `pplt.subplots` such as 'figwidth'.
    prof : bool or 'edges'
        Whether to overlay 1D profiles on 2D plots in off-diagonal axes. If
        'edges', only plot profiles in the left column and bottom row of
        the figure. This is a good option if not using diagonal subplots.
    prof_kws : dict
        Key word arguments passed to `image_profiless`.
    return_fig : bool
        Whether to return `fig` in addition to `axes`.
    return_mesh : bool
        Whether to also return a mesh from one of the pcolor plots. This is
        useful if you want to put a colorbar on the figure later.
    axes : proplot.gridspec
        Plot on an existing figure.
    **plot_kws
        Key word arguments passed to 2D plotting function.

    Returns
    -------
    axes : proplot.gridspec
        Array of subplot axes.
    Optional:
        fig : proplot.figure
            Proplot figure object.
        mesh : matplotlib.collections.QuadMesh
            Mesh from the latest call of `pcolormesh`. This is helpful if you want a
            global colorbar.
    """
    # Determine whether data is point cloud or image.
    n = data.ndim
    pts = False
    if n == 2:
        n = data.shape[1]
        pts = True

    # Parse arguments
    diag = diag_kind in ["line", "bar", "step", "linefilled", "stepfilled"]
    start = 1 if diag else 0
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault("color", "black")
    diag_kws.setdefault("lw", 1.0)
    if pts:
        if kind == "scatter":
            plot_kws.setdefault("s", 6)
            plot_kws.setdefault("c", "black")
            plot_kws.setdefault("marker", ".")
            plot_kws.setdefault("ec", "none")
            if "color" in plot_kws:
                plot_kws["c"] = plot_kws.pop("color")
            if "ms" in plot_kws:
                plot_kws["s"] = plot_kws.pop("ms")
            if "cmap" in plot_kws:
                plot_kws.pop("cmap")
        elif kind == "hist":
            plot_kws.setdefault("ec", "None")
            plot_kws.setdefault("process_kws", dict(mask_zero=True))
    else:
        plot_kws.setdefault("ec", "None")

    # Compute ax limits.
    if (not pts) and (coords is None):
        coords = [np.arange(s) for s in data.shape]
    if autolim_kws is None:
        autolim_kws = dict()
    if limits is None:
        if pts:
            limits = _vis_bunch.auto_limits(data, **autolim_kws)
        else:
            limits = [(c[0], c[-1]) for c in coords]

    # Create the figure.
    if axes is None:
        if fig_kws is None:
            fig_kws = dict()
        fig, axes = _setup_corner(n, diag, labels, limits, **fig_kws)
    else:
        # We are plotting on an existing figure. Expand the existing ax limits
        # based on our new data.
        if axes.shape[1] == n:
            old_limits = [axes[i, i].get_xlim() for i in range(n)]
        else:
            old_limits = [axes[-1, i].get_xlim() for i in range(n - 1)] + [axes[-1, 0].get_ylim()]
        limits = [
            (min(old_lim[0], lim[0]), max(old_lim[1], lim[1]))
            for old_lim, lim in zip(old_limits, limits)
        ]
        _set_corner_limits(axes, limits, diag=(axes.shape[1] == n))
        # Don't return fig because we do not have access to it.
        return_fig = False

    # Discrete points
    if pts:
        # Univariate plots
        bins = "auto"
        if "bins" in plot_kws:
            bins = plot_kws.pop("bins")
        edges, centers = [], []
        for i in range(n):
            heights, _edges = np.histogram(data[:, i], bins, limits[i], density=True)
            heights = heights / np.max(heights)
            _centers = utils.centers_from_edges(_edges)
            edges.append(_edges)
            centers.append(_centers)
            if diag:
                plot1d(_centers, heights, ax=axes[i, i], kind=diag_kind, **diag_kws)

        # Take random sample of points.
        idx = utils.random_selection(np.arange(data.shape[0]), samples)

        # Bivariate plots
        for ii, i in enumerate(range(start, axes.shape[0])):
            for j in range(ii + 1):
                ax = axes[i, j]
                if kind == "scatter":
                    ax.scatter(data[idx, j], data[idx, ii + 1], **plot_kws)
                elif kind == "hist":
                    _im, _, _ = np.histogram2d(
                        data[:, j], data[:, ii + 1], (edges[j], edges[ii + 1])
                    )
                    if prof == "edges":
                        profy = j == 0
                        profx = i == axes.shape[0] - 1
                    else:
                        profx = profy = prof
                    ax = _vis_image.plot2d(
                        _im,
                        coords=(centers[j], centers[ii + 1]),
                        ax=ax,
                        kind='pcolor',
                        profx=profx,
                        profy=profy,
                        prof_kws=prof_kws,
                        **plot_kws,
                    )
    # Multidimensional image
    else:
        # Bivariate plots
        for ii, i in enumerate(range(start, axes.shape[0])):
            for j in range(ii + 1):
                ax = axes[i, j]
                if prof == 'edges':
                    profy = j == 0
                    profx = i == axes.shape[0] - 1
                else:
                    profx = profy = prof
                _image = psi.project(data, (j, ii + 1))
                _image = _image / np.max(_image)
                ax, mesh = _vis_image.plot2d(
                    _image, coords=(coords[j], coords[ii + 1]),
                    ax=ax,
                    kind='pcolor',
                    profx=profx,
                    profy=profy,
                    prof_kws=prof_kws,
                    return_mesh=True,
                    **plot_kws,
                )
        # Univariate plots
        if diag:
            for i in range(n):
                profile = psi.project(data, i)
                profile = profile / np.max(profile)
                if 'process_kws' in plot_kws and 'fill_value' in plot_kws['process_kws']:
                    profile = np.ma.filled(profile, fill_value=plot_kws['fill_value'])
                plot1d(coords[i], profile, ax=axes[i, i], kind=diag_kind, **diag_kws)
    # Modify diagonal y axis limits.
    if diag:
        for i in range(n):
            axes[i, i].set_ylim(0, 1.0 / diag_height_frac)
    # Return items
    if return_fig:
        if return_mesh:
            return fig, axes, mesh
        return fig, axes
    return axes


# Interactive
# ------------------------------------------------------------------------------
