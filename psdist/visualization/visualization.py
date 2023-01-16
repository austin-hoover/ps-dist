"""Plotting routines for phase space distributions."""
from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt
import scipy.optimize

import psdist.image
import psdist.discrete
import psdist.utils as utils
import psdist.visualization.discrete as psv_discrete
import psdist.visualization.image as psv_image


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


def linear_fit(x, y):
    """Return (yfit, slope, intercept) from linear fit."""

    def fit(x, slope, intercept):
        return slope * x + intercept

    popt, pcov = scipy.optimize.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def plot1d(x, y, ax=None, offset=0.0, flipxy=False, kind="line", **kws):
    """Convenience function for one-dimensional line/step/bar plots."""
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


# The following functions set up a corner plot. It's probably better
# to have some object-oriented approach like in Seaborn?


def _corner_make_figure(n, diag, labels, limits=None, **fig_kws):
    if labels is None:
        labels = n * [""]
    nrows = ncols = n if diag else n - 1
    start = 1 if diag else 0
    fig_kws.setdefault("figwidth", 1.5 * nrows)
    fig_kws.setdefault("aligny", True)
    fig, axs = pplt.subplots(
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
                axs[i, j].axis("off")
    for ax, label in zip(axs[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axs[(1 if diag else 0) :, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(nrows):
        axs[:-1, i].format(xticklabels=[])
        axs[i, 1:].format(yticklabels=[])
    for ax in axs:
        ax.format(xspineloc="bottom", yspineloc="left")
    if diag:
        for i in range(n):
            axs[i, i].format(yspineloc="neither")
    if limits is not None:
        _set_corner_limits(axs, limits, diag=diag)
    axs.format(
        xtickminor=True, ytickminor=True, xlocator=("maxn", 3), ylocator=("maxn", 3)
    )
    return fig, axs


def _corner_set_limits(axs, limits, diag=False):
    for i in range(axs.shape[1]):
        axs[:, i].format(xlim=limits[i])
    start = int(diag)
    for i, lim in enumerate(limits[1:], start=start):
        axs[i, : (i + 1 - start)].format(ylim=lim)
    return axs


class CornerGrid:
    def __init__(
        self, 
        n=4, 
        diag=True, 
        labels=None, 
        limits=None, 
        diag_height_frac=0.6, 
        **fig_kws
    ):
        # Create figure.
        self.n = n
        self.diag = diag
        self.labels = labels
        self.limits = limits
        self.diag_height_frac = 0.6
        self.start = int(self.diag)
        if self.labels is None:
            self.labels = self.n * [""]
        self.nrows = self.ncols = self.n
        if not self.diag:
            self.nrows = self.nrows - 1
            self.ncols = self.ncols - 1
        self.fig_kws = fig_kws
        self.fig_kws.setdefault("figwidth", 1.5 * self.nrows)
        self.fig_kws.setdefault("aligny", True)
        self.fig, self.axs = pplt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            sharex=1,
            sharey=1,
            spanx=False,
            spany=False,
            **self.fig_kws,
        )

        # Collect diagonal/off-diagonal subplots.
        self.axs_diag, self.axs_offdiag = [], []
        self.axis_diag, self.axis_offdiag = [], []
        if self.diag:
            for i in range(self.n):
                self.axs_diag.append(self.axs[i, i])
                self.axis_diag.append(i)
            for i in range(1, self.n):
                for j in range(i):
                    self.axs_offdiag.append(self.axs[i, j])
                    self.axis_offdiag.append((j, i))
        else:
            for i in range(self.n - 1):
                for j in range(i + 1):
                    self.axs_offdiag.append(self.axs[i, j])
                    self.axis_offdiag.append((j, i + 1))

        # 
        for i in range(self.nrows):
            for j in range(self.ncols):
                if j > i:
                    self.axs[i, j].axis('off')
        for ax, label in zip(self.axs[-1, :], self.labels):
            ax.format(xlabel=label)
        for ax, label in zip(self.axs[self.start :, 0], self.labels[1:]):
            ax.format(ylabel=label)
        for i in range(self.nrows):
            self.axs[:-1, i].format(xticklabels=[])
            self.axs[i, 1:].format(yticklabels=[])
        for ax in self.axs:
            ax.format(xspineloc='bottom', yspineloc='left')
        for ax in self.axs_diag:
            ax.format(yspineloc='neither')
        self.axs.format(xtickminor=True, ytickminor=True, 
                        xlocator=('maxn', 3), ylocator=('maxn', 3))
        self.set_limits(limits)

    def get_limits(self):
        if self.diag:
            limits = [ax.get_xlim() for ax in self.axs_diag]
        else:
            limits = [self.axs[-1, i].get_xlim() for i in range(self.n - 1)]
            limits = limits + [self.axs[-1, 0].get_ylim()]
        return limits

    def set_limits(self, limits=None, expand=False):
        if limits is not None:
            if expand:
                limits = np.array(limits)
                limits_old = np.array(self.get_limits())
                mins = np.minimum(limits[:, 0], limits_old[:, 0])
                maxs = np.maximum(limits[:, 1], limits_old[:, 1])
                limits = list(zip(mins, maxs))
            for i in range(self.axs.shape[1]):
                self.axs[:, i].format(xlim=limits[i])
            for i, lim in enumerate(limits[1:], start=self.start):
                self.axs[i, : (i + 1 - self.start)].format(ylim=lim)
        self.limits = self.get_limits()

    def scale_diag_ylim(self, factor):
        for ax in self.axs_diag:
            ymin, ymax = ax.get_ylim()
            ax.format(ylim=(0.0, factor * ymax))

    def plot_image(
        self,
        f,
        coords=None,
        prof_edge_only=False,
        update_limits=True,
        diag_kws=None,
        **kws
    ):
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault('color', 'black')
        diag_kws.setdefault('lw', 1.0)
        diag_kws.setdefault('kind', 'step')
        kws.setdefault('kind', 'pcolor')
        kws.setdefault('profx', False)
        kws.setdefault('profy', False)

        if coords is None:
            coords = [np.arange(f.shape[i]) for i in range(f.ndim)]

        if update_limits:
            edges = [utils.edges_from_centers(c) for c in coords]
            limits = [(np.min(e), np.max(e)) for e in edges]
            self.set_limits(limits, expand=update_limits)

        # Univariate plots.
        for ax, axis in zip(self.axs_diag, self.axis_diag):
            profile = psdist.image.project(f, axis=axis)
            profile = profile / np.max(profile)
            plot1d(coords[axis], profile, ax=ax, **diag_kws)

        # Bivariate plots.
        profx, profy = [kws.pop(key) for key in ('profx', 'profy')]
        for ax, axis in zip(self.axs_offdiag, self.axis_offdiag):
            if prof_edge_only:
                if profx: 
                    kws['profx'] = axis[1] == self.n - 1
                if profy:
                    kws['profy'] = axis[0] == 0
            _f = psdist.image.project(f, axis=axis)
            _f = _f / np.max(_f)
            _coords = [coords[i] for i in axis]
            psv_image.plot2d(_f, coords=_coords, ax=ax, **kws)

        # Modify diagonal y axis limits.
        if self.diag_height_frac is not None:
            self.scale_diag_ylim(1.0 / self.diag_height_frac)

    def plot_discrete(
        self,
        X,
        diag_kws=None,
        limits=None,
        autolim_kws=None,
        prof_edge_only=False,
        update_limits=True,
        **kws
    ):
        n = X.shape[1]
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault('color', 'black')
        diag_kws.setdefault('lw', 1.0)
        diag_kws.setdefault('kind', 'step')
        kws.setdefault('kind', 'hist')
        kws.setdefault('profx', False)
        kws.setdefault('profy', False)

        if limits is None:
            if autolim_kws is None:
                autolim_kws = dict()
            limits = psv_discrete.auto_limits(X, **autolim_kws)
        self.set_limits(limits, expand=update_limits)

        # Univariate plots. Remember histogram bins and use them for 2D histograms.
        bins = 'auto'
        if 'bins' in kws:
            bins = kws.pop('bins')
        edges = []
        for ax, axis in zip(self.axs_diag, self.axis_diag):
            heights, _edges = np.histogram(X[:, axis], bins, limits[axis], density=True)
            edges.append(_edges)
            plot1d(
                utils.centers_from_edges(_edges), 
                heights / np.max(heights), 
                ax=ax, 
                **diag_kws
            )

        # Bivariate plots:
        profx, profy = [kws.pop(key) for key in ('profx', 'profy')]
        for ax, axis in zip(self.axs_offdiag, self.axis_offdiag):
            if prof_edge_only:
                if profx: 
                    kws['profx'] = axis[1] == self.n - 1
                if profy:
                    kws['profy'] = axis[0] == 0
            if kws['kind'] == 'hist':
                kws['bins'] = [edges[axis[0]], edges[axis[1]]]
            psv_discrete.plot2d(X[:, axis], ax=ax, **kws)
