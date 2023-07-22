import numpy as np
import proplot as pplt

import psdist.image
import psdist.cloud
import psdist.utils
import psdist.visualization.cloud as vis_cloud
import psdist.visualization.image as vis_image
from psdist.visualization.visualization import plot_profile
from psdist.visualization.visualization import scale_profile


class JointGrid:
    """Grid for joint plots.

    Attributes
    -----------
    fig : proplot.figure.Figure
        The main figure.
    ax : proplot.gridspec.SubplotGrid
        The main axis.
    ax_marg_x, ax_marg_y : proplot.gridspec.SubplotGrid
        The marginal (panel) axes on the top and right.
    """

    def __init__(
        self,
        marg_kws=None,
        marg_fmt_kws=None,
        marg_fmt_kws_x=None,
        marg_fmt_kws_y=None,
        **fig_kws,
    ):
        """Constructor.

        marg_kws : dict
            Key word arguments for `ax.panel`.
        marg_fmt_kws, marg_fmt_kws_x, marg_fmt_kws_y : dict
            Key word arguments for `ax.format` for each of the marginal (panel) axs.
        **fig_kws
            Key word arguments passed to `proplot.subplots`.
        """
        self.fig, self.ax = pplt.subplots(**fig_kws)
        if marg_kws is None:
            marg_kws = dict()
        if marg_fmt_kws is None:
            marg_fmt_kws = dict()
        if marg_fmt_kws_x is None:
            marg_fmt_kws_x = dict()
        if marg_fmt_kws_y is None:
            marg_fmt_kws_y = dict()

        marg_fmt_kws_x.setdefault("xspineloc", "bottom")
        marg_fmt_kws_x.setdefault("yspineloc", "left")
        marg_fmt_kws_y.setdefault("xspineloc", "bottom")
        marg_fmt_kws_y.setdefault("yspineloc", "left")

        self.ax_marg_x = self.ax.panel("t", **marg_kws)
        self.ax_marg_y = self.ax.panel("r", **marg_kws)
        self.marg_axs = [self.ax_marg_x, self.ax_marg_y]
        for ax in self.marg_axs:
            ax.format(**marg_fmt_kws)
        self.marg_axs[0].format(**marg_fmt_kws_x)
        self.marg_axs[1].format(**marg_fmt_kws_y)

    def get_default_marg_kws(self, marg_kws=None):
        if marg_kws is None:
            marg_kws = dict()
        marg_kws.setdefault("color", "black")
        marg_kws.setdefault("kind", "step")
        marg_kws.setdefault("lw", 1.0)
        marg_kws.setdefault("scale", "density")
        return marg_kws

    def plot_cloud(self, X, marg_hist_kws=None, marg_kws=None, **kws):
        """Plot a 2D point cloud.

        Parameters
        ----------
        X : ndarray, shape (k, 2)
            Coordinates of k points in 2-dimensional space.
        marg_hist_kws : dict
            Key word arguments passed to `np.histogram` for 1D histograms.
        marg_kws : dict
            Key word arguments passed to `visualization.plot_profile`.
        **kws
            Key word arguments passed to `visualization.image.plot2d.`
        """
        marg_kws = self.get_default_marg_kws(marg_kws)
        if marg_hist_kws is None:
            marg_hist_kws = dict()
        marg_hist_kws.setdefault("bins", "auto")
        kws.setdefault("kind", "hist")
        if kws["kind"] == "hist":
            kws.setdefault("mask", True)
            kws.setdefault("bins", marg_hist_kws["bins"])
        if kws["kind"] != "scatter":
            kws.setdefault("colorbar_kw", dict())
            kws["colorbar_kw"].setdefault("pad", 2.0)
        for axis in range(2):
            profile, edges = np.histogram(X[:, axis], **marg_hist_kws)
            plot_profile(
                profile=profile,
                edges=edges,
                ax=self.marg_axs[axis],
                orientation=("horizontal" if bool(axis) else "vertical"),
                **marg_kws,
            )
        vis_cloud.plot2d(X, ax=self.ax, **kws)

    def plot_image(self, f, coords=None, marg_kws=None, **kws):
        """Plot a 2D image.

        Parameters
        ----------
        f : ndarray
            A d-dimensional image.
        coords : list[ndarray]
            Coordinates along each dimension of `f`.
        marg_kws : dict
            Key word arguments passed to `visualization.plot_profile`.
        **kws
            Key word arguments passed to `visualization.image.plot2d.`
        """
        marg_kws = self.get_default_marg_kws(marg_kws)
        kws.setdefault("colorbar_kw", dict())
        kws["colorbar_kw"].setdefault("pad", 2.0)
        if coords is None:
            coords = [np.arange(f.shape[axis]) for axis in range(f.ndim)]
        for axis in range(2):
            plot_profile(
                profile=psdist.image.project(f, axis),
                coords=coords[axis],
                ax=self.marg_axs[axis],
                orientation=("horizontal" if bool(axis) else "vertical"),
                **marg_kws,
            )
        return vis_image.plot2d(f, coords=coords, ax=self.ax, **kws)

    def colorbar(self, mappable, **kws):
        """Add a colorbar."""
        kws.setdefault("loc", "r")
        kws.setdefault("pad", 2.0)
        self.fig.colorbar(mappable, **kws)


class CornerGrid:
    """Grid for corner plots.

    Attributes
    ----------
    fig : proplot.figure.Figure
        The main figure.
    axs : proplot.gridspec.SubplotGrid
        The subplot axes.
    diag_axs : list[proplot.gridspec.SubplotGrid]
        The axes for diagonal (univariate) subplots. Can be empty.
    offdiag_axs : list[proplot.gridspec.SubplotGrid]
        The axes for off-diagonal (bivariate) subplots.
    diag_indices : list[int]
        The index of the dimension plotted on each diagonal subplot.
    offdiag_indices : list[2-tuple of int]
        Indices of the dimensions plotted on each off-diagonal subplot.
    """

    def __init__(
        self,
        d=4,
        diag=True,
        diag_shrink=1.0,
        diag_rspine=False,
        diag_share=True,
        limits=None,
        labels=None,
        corner=True,
        **fig_kws,
    ):
        """
        Parameters
        ----------
        d : int
            The number of rows/columns.
        diag : bool
            Whether to include diagonal subplots (univariate plots). If False,
            we have an (n - 1) x (n - 1) grid instead of an n x n grid.
        diag_shrink : float in range [0, 1]
            Scales the maximum y value of the diagonal profile plots.
        diag_rspine : bool
            Whether to include right spine on diagonal subplots (if `corner`).
        diag_share : bool
            Whether to share diagonal axis limits; i.e., whether we can compare
            the areas under the diagonal profile plots.
        limits : list[tuple], length n
            The (min, max) for each dimension. (These can be set later.)
        labels : list[str]
            The label for each dimension. (These can be set later.)
        corner : bool
            Whether to hide the upper-triangular subplots.
        **fig_kws
            Key word arguments passed to `pplt.subplots()`.
        """
        # Create figure.
        self.new = True
        self.d = self.nrows = self.ncols = d
        self.corner = corner
        self.diag = diag
        self.diag_shrink = diag_shrink
        self.diag_rspine = diag_rspine
        self.diag_share = diag_share
        self.diag_ymin = None
        self.diag_yscale = self.d * [None]
        if not self.diag:
            self.nrows = self.nrows - 1
            self.ncols = self.ncols - 1
        self.fig_kws = fig_kws
        self.fig_kws.setdefault("figwidth", 1.5 * self.nrows)
        self.fig_kws.setdefault("aligny", True)
        self.fig, self.axs = pplt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            spanx=False,
            spany=False,
            sharex=False,
            sharey=False,
            **self.fig_kws,
        )
        # Collect diagonal/off-diagonal subplots and indices.
        self.diag_axs = []
        self.offdiag_axs = []
        self.offdiag_axs_u = []
        self.diag_indices = []
        self.offdiag_indices = []
        self.offdiag_indices_u = []
        if self.diag:
            for i in range(self.d):
                self.diag_axs.append(self.axs[i, i])
                self.diag_indices.append(i)
            for i in range(1, self.d):
                for j in range(i):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_axs_u.append(self.axs[j, i])
                    self.offdiag_indices.append((j, i))
                    self.offdiag_indices_u.append((i, j))
        else:
            for i in range(self.d - 1):
                for j in range(i + 1):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_indices.append((j, i + 1))

        # Set limits and labels.
        self.limits = limits
        if limits is not None:
            self.set_limits(limits)
        self.labels = labels
        if labels is not None:
            self.set_labels(labels)

        # Formatting
        if self.corner or not self.diag:
            for i in range(self.nrows):
                for j in range(self.ncols):
                    if j > i:
                        self.axs[i, j].axis("off")
        self.axs[:-1, :].format(xticklabels=[])
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[i, j]
                if i != self.nrows - 1:
                    ax.format(xticklabels=[])
                if j != 0:
                    if not (i == j and self.diag_rspine and self.corner and self.diag):
                        ax.format(yticklabels=[])
        self.axs.format(xspineloc="bottom", yspineloc="left")
        if self.corner:
            if self.diag_rspine:
                self.format_diag(yspineloc="right")
            else:
                self.format_diag(yspineloc="neither")
        self.axs.format(
            xtickminor=True, ytickminor=True, xlocator=("maxn", 3), ylocator=("maxn", 3)
        )
        self.set_diag_scale("linear")

    def format_diag(self, **kws):
        """Format diagonal subplots."""
        for ax in self.diag_axs:
            ax.format(**kws)
        if not self.corner:
            for ax in self.diag_axs[1:]:
                ax.format(yticklabels=[])
        self._fake_diag_yticks()

    def format_offdiag(self, **kws):
        """Format off-diagonal subplots."""
        for ax in [self.offdiag_axs + self.offdiag_axs_u]:
            ax.format(**kws)

    def get_labels(self):
        """Return the dimension labels."""
        if self.diag:
            labels = [ax.get_xlabel() for ax in self.diag_axs]
        else:
            labels = [self.axs[-1, i].get_xlabel() for i in range(self.d - 1)]
            labels = labels + [self.axs[-1, 0].get_ylabel()]
        return labels

    def set_labels(self, labels):
        """Set the dimension labels."""
        for ax, label in zip(self.axs[-1, :], labels):
            ax.format(xlabel=label)
        for ax, label in zip(self.axs[int(self.diag) :, 0], labels[1:]):
            ax.format(ylabel=label)
        if self.diag and not self.corner:
            self.axs[0, 0].format(ylabel=labels[0])
        self.labels = labels

    def get_limits(self):
        """Return the plot limits."""
        if self.diag:
            limits = [ax.get_xlim() for ax in self.diag_axs]
        else:
            limits = [self.axs[-1, i].get_xlim() for i in range(self.d - 1)]
            limits = limits + [self.axs[-1, 0].get_ylim()]
        return limits

    def set_limits(self, limits=None, expand=False):
        """Set the plot limits.

        Parameters
        ----------
        limits : list[tuple], length n
            The (min, max) for each dimension.
        expand : bool
            If True, compare the proposed limits to the existing limits, expanding
            if the new limits are wider.
        """
        if limits is not None:
            if expand:
                limits = np.array(limits)
                limits_old = np.array(self.get_limits())
                mins = np.minimum(limits[:, 0], limits_old[:, 0])
                maxs = np.maximum(limits[:, 1], limits_old[:, 1])
                limits = list(zip(mins, maxs))
            for (i, j), ax in zip(self.offdiag_indices, self.offdiag_axs):
                ax.format(ylim=limits[j])
            for i in range(self.ncols):
                self.axs[:, i].format(xlim=limits[i])
        self.limits = self.get_limits()

    def get_default_diag_kws(self, diag_kws=None):
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("lw", 1.0)
        diag_kws.setdefault("kind", "step")
        diag_kws.setdefault("scale", "density")
        return diag_kws

    def _fake_diag_yticks(self):
        """The yticks on the (0, 0) subplot correspond to the other subplots in the row.

        Source: pandas.plotting.scatterplot_matrix.
        """
        if self.corner or not self.diag:
            return
        limits = self.limits
        if limits is None:
            limits = self.get_limits()

        lim1 = limits[0]
        locs = self.axs[0, 0].xaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = self.axs[0, 0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        self.axs[0, 0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            locs = locs.astype(int)
        self.axs[0, 0].yaxis.set_ticklabels(locs)

    def _force_non_negative_diag_ymin(self):
        """Force diagonal ymins to be at least zero."""
        if any([ax.get_ylim()[0] < 0.0 for ax in self.diag_axs]):
            self.format_diag(ylim=0.0)

    def set_diag_scale(self, scale="linear", pad=0.05):
        """Set diagonal y axis scale.

        Parameters
        ----------
        scale : {"linear", "log"}
            If "linear", scale runs from 0 to 1. If "log", scale runs from half the
            minimum plotted value to 1.
        """
        if scale == "linear":
            ymin = 0.0
            ymax = 1.0 / self.diag_shrink
            delta = ymax - ymin
            ymax = ymin + delta * (1.0 + pad)
            self.format_diag(yscale="linear", yformatter="auto", ymin=ymin, ymax=ymax)
        elif scale == "log":
            ymin = self.diag_ymin
            ymax = 1.0
            log_ymin = np.log10(ymin)
            log_ymax = np.log10(ymax)
            log_delta = log_ymax - log_ymin
            log_delta = log_delta / self.diag_shrink
            ymax = 10.0 ** (log_ymin + log_delta * (1.0 + pad))
            self.format_diag(yscale="log", yformatter="log", ymin=ymin, ymax=ymax)

    def _plot_diag(self, get_data, **kws):
        # Compute one-dimensional probabiliy density functions.
        if "scale" in kws:
            kws.pop("scale")
        profiles, edges_list = [], []
        for axis in range(self.d):
            profile, edges = get_data(axis)
            profile = scale_profile(profile, edges=edges, scale="density")
            profiles.append(profile)
            edges_list.append(edges)
        # Update scaling factor for each profile, taking into account all existing plots.
        for axis in range(self.d):
            if self.diag_yscale[axis] is None:
                self.diag_yscale[axis] = -np.inf
        if self.diag_share:
            max_value = max([np.max(profile) for profile in profiles])
            for axis in range(self.d):
                self.diag_yscale[axis] = max(self.diag_yscale[axis], max_value)
        else:
            for axis in range(self.d):
                self.diag_yscale[axis] = max(
                    self.diag_yscale[axis], np.max(profiles[axis])
                )
        # Plot the profiles.
        for axis in range(self.d):
            if self.diag_yscale[axis]:
                profiles[axis] = profiles[axis] / self.diag_yscale[axis]
            plot_profile(
                profiles[axis], edges=edges_list[axis], ax=self.diag_axs[axis], **kws
            )
        # Compute min positive value for log scaling.
        for axis in range(self.d):
            if not self.diag_ymin:
                self.diag_ymin = np.inf
            self.diag_ymin = min(self.diag_ymin, np.min(profile[profile > 0.0]))

    def plot_image(
        self,
        f,
        coords=None,
        prof_edge_only=False,
        lower=True,
        upper=True,
        diag=True,
        update_limits=True,
        diag_kws=None,
        **kws,
    ):
        """Plot an image.

        Parameters
        ----------
        f : ndarray
            A d-dimensional image.
        coords : list[ndarray]
            Coordinates along each axis of the grid (if `data` is an image).
        prof_edge_only : bool
            If plotting profiles on top of images (on off-diagonal subplots), whether
            to plot x profiles only in bottom row and y profiles only in left column.
        lower, upper, diag : bool
            Whether to plot on the lower triangular, upper triangular, and/or diagonal subplots.
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `visualization.plot_profile`.
        **kws
            Key word arguments pass to `visualization.image.plot2d`
        """
        diag_kws = self.get_default_diag_kws(diag_kws)
        kws.setdefault("kind", "pcolor")
        kws.setdefault("profx", False)
        kws.setdefault("profy", False)
        kws.setdefault("process_kws", dict())
        kws["process_kws"].setdefault("norm", "max")

        if coords is None:
            coords = [np.arange(f.shape[i]) for i in range(f.ndim)]
        edges = [psdist.utils.edges_from_centers(c) for c in coords]

        if update_limits:
            limits = [(np.min(e), np.max(e)) for e in edges]
            self.set_limits(limits, expand=(not self.new))
        self.new = False

        # Univariate plots.
        if self.diag and diag:
            self._plot_diag(
                lambda axis: (psdist.image.project(f, axis=axis), edges[axis]),
                **diag_kws,
            )

        # Bivariate plots.
        profx, profy = [kws.pop(key) for key in ("profx", "profy")]
        if lower:
            for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
                if prof_edge_only:
                    if profx:
                        kws["profx"] = axis[1] == self.d - 1
                    if profy:
                        kws["profy"] = axis[0] == 0
                vis_image.plot2d(
                    psdist.image.project(f, axis=axis),
                    coords=[coords[k] for k in axis],
                    ax=ax,
                    **kws,
                )
        if not self.corner and upper:
            for ax, axis in zip(self.offdiag_axs_u, self.offdiag_indices_u):
                _f = psdist.image.project(f, axis=axis)
                _f = _f / np.max(_f)
                _coords = [coords[k] for k in axis]
                vis_image.plot2d(_f, coords=_coords, ax=ax, **kws)
        self._cleanup()

    def plot_cloud(
        self,
        X,
        limits=None,
        bins="auto",
        autolim_kws=None,
        prof_edge_only=False,
        lower=True,
        upper=True,
        diag=True,
        update_limits=True,
        diag_kws=None,
        **kws,
    ):
        """Plot a point cloud.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Coordinates of n points in d-dimensional space.
        limits : list[tuple], length d
            The (min, max) axis limits.
        bins : 'auto', int, list[int]
            The number of bins along each dimension (if plot type requires histogram
            computation). If int or 'auto', applies to all dimensions. Currently
            the histogram is computed with limits based on the data min/max, not
            the plot limits.
        prof_edge_only : bool
            If plotting profiles on top of images (on off-diagonal subplots), whether
            to plot x profiles only in bottom row and y profiles only in left column.
        lower, upper, diag : bool
            Whether to plot on the lower triangular, upper triangular, and/or diagonal subplots.
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `visualization.plot_profile`.
        **kws
            Key word arguments pass to `visualization.cloud.plot2d`
        """
        diag_kws = self.get_default_diag_kws(diag_kws)
        kws.setdefault("kind", "hist")
        kws.setdefault("profx", False)
        kws.setdefault("profy", False)

        if limits is None:
            if autolim_kws is None:
                autolim_kws = dict()
            autolim_kws.setdefault("pad", 0.1)
            limits = vis_cloud.auto_limits(X, **autolim_kws)
        if update_limits:
            self.set_limits(limits, expand=(not self.new))
        limits = self.get_limits()
        self.new = False

        if not psdist.utils.array_like(bins):
            bins = X.shape[1] * [bins]

        # Compute histogram bin edges.
        edges = []
        for axis in range(self.d):
            if psdist.utils.array_like(bins[axis]):
                edges.append(bins[axis])
            else:
                edges.append(
                    np.histogram_bin_edges(X[:, axis], bins[axis], limits[axis])
                )

        # Univariate plots.
        if self.diag and diag:
            self._plot_diag(
                lambda axis: np.histogram(X[:, axis], edges[axis]), **diag_kws
            )

        # Bivariate plots:
        profx, profy = [kws.pop(key) for key in ["profx", "profy"]]
        if lower:
            for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
                if prof_edge_only:
                    if profx:
                        kws["profx"] = axis[1] == self.d - 1
                    if profy:
                        kws["profy"] = axis[0] == 0
                if kws["kind"] in ["hist", "contour", "contourf"]:
                    kws["bins"] = [edges[axis[0]], edges[axis[1]]]
                vis_cloud.plot2d(X[:, axis], ax=ax, **kws)
        if upper and not self.corner:
            for ax, axis in zip(self.offdiag_axs_u, self.offdiag_indices_u):
                if kws["kind"] in ["hist", "contour", "contourf"]:
                    kws["bins"] = [edges[axis[0]], edges[axis[1]]]
                vis_cloud.plot2d(X[:, axis], ax=ax, **kws)
        self._cleanup()

    def _cleanup(self):
        self._fake_diag_yticks()
        self._force_non_negative_diag_ymin()


class SliceGrid:
    """Grid for slice matrix plots (https://arxiv.org/abs/2301.04178).

    This plot is used to visualize four dimensions of a distribution f(x1, x2, x3, x4).

    The main panel is an nrows x ncols grid that shows f(x1, x2 | x3, x4) -- the
    x1-x2 distribution for a planar slice in x3-x4. Each subplot corresponds to a
    different location in the x3-x4 plane.

    The following is only included if `marginals` is True:

        The bottom panel shows the marginal 3D distribution f(x1, x2 | x3).

        The right panel shows the marginal 3D distribution f(x1, x2 | x4).

        The bottom right subplot shows the full projection f(x1, x2).

        The lone subplot on the bottom right shows f(x1, x2)l, the full projection
        onto the x1-x2 plane.

    Attributes
    ----------
    fig : proplot.figure.Figure
        The main figure.
    axs : proplot.gridspec.SubplotGrid
        The subplot axes.
    _axs : proplot.figure.Figure
        The subplot axes on the main panel.
    _axs_marg_x, _axs_marg_y, _axs_marg_xy : proplot.gridspec.SubplotGrid
        The subplot axes on the marginal panels.
    """

    def __init__(
        self,
        nrows=9,
        ncols=9,
        space=0.0,
        gap=2.0,
        marginals=True,
        annotate=True,
        annotate_kws_view=None,
        annotate_kws_slice=None,
        slice_label_height=0.22,
        **fig_kws,
    ):
        """Constructor.

        nrows, ncols : int
            The number of rows/colums in the figure.
        space : float
            Spacing between subplots.
        gap : float
            Gap between main and marginal panels.
        marginals : bool
            Whether to include the marginal panels. If they are not included, we just
            have an nrows x ncols grid.
        annotate : bool
            Whether to add dimension labels/arrows to the figure.
        annotate_kws_view, annotate_kws_slice : dict
            Key word arguments for figure text. The 'view' key words are for the view
            dimension labels; they are printed on top of one of the subplots. The
            'slice' key words are for the slice dimension labels; they are printed
            on the sides of the figure, between the main and marginal panels.
        slice_label_height : float
            Tweaks the position of slice labels. Need a better way to handle this.
        **fig_kws
            Key word arguments for `pplt.subplots`.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.space = space
        self.gap = gap
        self.marginals = marginals
        self.annotate = annotate
        self.slice_label_height = slice_label_height
        self.fig_kws = fig_kws
        self.axis_slice = None
        self.axis_view = None
        self.ind_slice = None

        self.annotate_kws_view = annotate_kws_view
        if self.annotate_kws_view is None:
            self.annotate_kws_view = dict()
        self.annotate_kws_view.setdefault("color", "black")
        self.annotate_kws_view.setdefault("xycoords", "axes fraction")
        self.annotate_kws_view.setdefault("horizontalalignment", "center")
        self.annotate_kws_view.setdefault("verticalalignment", "center")

        self.annotate_kws_slice = annotate_kws_slice
        if self.annotate_kws_slice is None:
            self.annotate_kws_slice = dict()
        self.annotate_kws_slice.setdefault("color", "black")
        self.annotate_kws_slice.setdefault("xycoords", "axes fraction")
        self.annotate_kws_slice.setdefault("horizontalalignment", "center")
        self.annotate_kws_slice.setdefault("verticalalignment", "center")
        self.annotate_kws_slice.setdefault(
            "arrowprops", dict(arrowstyle="->", color="black")
        )

        fig_kws.setdefault("figwidth", 8.5 * (ncols / 13.0))
        fig_kws.setdefault("share", True)
        fig_kws["ncols"] = ncols + 1 if marginals else ncols
        fig_kws["nrows"] = nrows + 1 if marginals else nrows
        hspace = nrows * [space]
        wspace = ncols * [space]
        if marginals:
            hspace[-1] = wspace[-1] = gap
        else:
            hspace = hspace[:-1]
            wspace = wspace[:-1]
        fig_kws["hspace"] = hspace
        fig_kws["wspace"] = wspace

        self.fig, self.axs = pplt.subplots(**fig_kws)

        self._axs = self.axs[:-1, :-1]
        self._axs_marg_x = []
        self._axs_marg_y = []
        if self.marginals:
            self._axs_marg_x = self.axs[-1, :]
            self._axs_marg_y = self.axs[:, -1]
        self._ax_marg_xy = self.axs[-1, -1]

    def _annotate(
        self,
        labels=None,
        slice_label_height=0.22,
        annotate_kws_view=None,
        annotate_kws_slice=None,
    ):
        """Add dimension labels and arrows."""
        # Label the view dimensions.
        for i, xy in enumerate([(0.5, 0.13), (0.12, 0.5)]):
            self.axs[0, 0].annotate(labels[i], xy=xy, **self.annotate_kws_view)

        # Label the slice dimensions. Print dimension labels with arrows like this:
        # "<----- x ----->" on the bottom and right side of the main panel.
        arrow_length = 2.5  # arrow length
        text_length = 0.15  # controls space between dimension label and start of arrow
        i = -1 - int(self.marginals)
        anchors = (self.axs[i, self.ncols // 2], self.axs[self.nrows // 2, i])
        anchors[0].annotate(
            labels[2], xy=(0.5, -slice_label_height), **annotate_kws_slice
        )
        anchors[1].annotate(
            labels[3], xy=(1.0 + slice_label_height, 0.5), **annotate_kws_slice
        )
        for arrow_direction in (1.0, -1.0):
            anchors[0].annotate(
                "",
                xy=(0.5 + arrow_direction * arrow_length, -slice_label_height),
                xytext=(0.5 + arrow_direction * text_length, -slice_label_height),
                **annotate_kws_slice,
            )
            anchors[1].annotate(
                "",
                xy=(1.0 + slice_label_height, 0.5 + arrow_direction * arrow_length),
                xytext=(1.0 + slice_label_height, 0.5 + arrow_direction * text_length),
                **annotate_kws_slice,
            )

    def get_ind_slice(self):
        """Return slice indices from latest plot call."""
        return self.ind_slice

    def get_axis_slice(self):
        """Return slice axis from latest plot call."""
        return self.axis_slice

    def get_axis_view(self):
        """Return view axis from latest plot call."""
        return self.axis_view

    def set_limits(self, limits):
        """Set the plot limits."""
        for ax in self.axs:
            ax.format(xlim=limits[0], ylim=limits[1])

    def plot_image(
        self,
        f,
        coords=None,
        labels=None,
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
        **kws,
    ):
        """Plot a d-dimensional image.

        Parameters
        ----------
        f : ndarray
            An d-dimensional image.
        coords : list[ndarray]
            Coordinates along each axis of the grid (if `data` is an image).
        labels : list[str], length n
            Label for each dimension.
        axis_view, axis_slice : 2-tuple of int
            The axis to view (plot) and to slice.
        pad : int, float, list
            This determines the start/stop indices along the sliced dimensions. If
            0, space the indices along axis `i` uniformly between 0 and `f.shape[i]`.
            Otherwise, add a padding equal to `int(pad[i] * f.shape[i])`. So, if
            the shape=10 and pad=0.1, we would start from 1 and end at 9.
        debug : bool
            Whether to print debugging messages.
        **kws
            Key word arguments pass to `visualization.image.plot2d`
        """
        # Setup
        # -----------------------------------------------------------------------
        if f.ndim < 4:
            raise ValueError(f"f.ndim = {f.ndim} < 4")
        if coords is None:
            coords = [np.arange(s) for s in f.shape]
        self.axis_view = axis_view
        self.axis_slice = axis_slice

        # Compute 4D/3D/2D projections.
        _f = psdist.image.project(f, axis_view + axis_slice)
        _fx = psdist.image.project(f, axis_view + axis_slice[:1])
        _fy = psdist.image.project(f, axis_view + axis_slice[1:])
        _fxy = psdist.image.project(f, axis_view)

        # Compute new coords and labels.
        _coords = [coords[i] for i in axis_view + axis_slice]
        _labels = None
        if labels is not None:
            _labels = [labels[i] for i in axis_view + axis_slice]

        # Select slice indices.
        if type(pad) in [float, int]:
            pad = len(axis_slice) * [pad]
        ind_slice = []
        for i, nsteps, _pad in zip(axis_slice, [self.nrows, self.ncols], pad):
            start = int(_pad * f.shape[i])
            stop = f.shape[i] - 1 - start
            if stop - start == nsteps - 1:
                ind_slice.append(np.arange(nsteps))
            elif (stop - start) < nsteps:
                raise ValueError(f"f.shape[{i}] < number of slice indices requested.")
            ind_slice.append(np.linspace(start, stop, nsteps).astype(int))
        ind_slice = tuple(ind_slice)
        self.ind_slice = ind_slice

        if debug:
            print("Slice indices:")
            for ind in ind_slice:
                print(ind)

        # Slice the 4D projection. The axes order was already handled by `project`;
        # the first two axes are the view axes and the last two axes are the
        # slice axes.
        axis_view = (0, 1)
        axis_slice = (2, 3)
        idx = 4 * [slice(None)]
        for axis, ind in zip(axis_slice, ind_slice):
            idx[axis] = ind
            _f = _f[tuple(idx)]
            idx[axis] = slice(None)

        # Slice the 3D projections.
        _fx = _fx[:, :, ind_slice[0]]
        _fy = _fy[:, :, ind_slice[1]]

        # Slice coords.
        for i, ind in zip(axis_slice, ind_slice):
            _coords[i] = _coords[i][ind]

        # Normalize each distribution.
        _f = psdist.image.process(_f, norm="max")
        _fx = psdist.image.process(_fx, norm="max")
        _fy = psdist.image.process(_fy, norm="max")
        _fxy = psdist.image.process(_fxy, norm="max")

        if debug:
            print("_f.shape =", _f.shape)
            print("_fx.shape =", _fx.shape)
            print("_fy.shape =", _fy.shape)
            print("_fxy.shape =", _fxy.shape)
            for i in range(_f.ndim):
                assert _f.shape[i] == len(_coords[i])

        # Add dimension labels to the figure.
        if self.annotate and _labels is not None:
            self._annotate(
                labels=_labels,
                slice_label_height=self.slice_label_height,
                annotate_kws_view=self.annotate_kws_view,
                annotate_kws_slice=self.annotate_kws_slice,
            )

        # Plotting
        # -----------------------------------------------------------------------
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[self.nrows - 1 - i, j]
                idx = psdist.image.slice_idx(
                    _f.ndim, axis=axis_slice, ind=[(j, j + 1), (i, i + 1)]
                )
                vis_image.plot2d(
                    psdist.image.project(_f[idx], axis_view),
                    coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
        if self.marginals:
            for i, ax in enumerate(reversed(self.axs[:-1, -1])):
                vis_image.plot2d(
                    _fy[:, :, i],
                    coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
            for i, ax in enumerate(self.axs[-1, :-1]):
                vis_image.plot2d(
                    _fx[:, :, i],
                    [_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
            vis_image.plot2d(
                _fxy,
                coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                ax=self.axs[-1, -1],
                **kws,
            )

    def _plot_cloud(
        self,
        X,
        labels=None,
        bins_slice="auto",
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
        autolim_kws=None,
        **kws,
    ):
        """Plot a point cloud.

        NOTE: this is not currently working... for the time being, it is recommended
        to generate a 4D histogram and then call `plot_image`.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Coordinates of n points in d-dimensional space.
        labels : list[str], length d
            Label for each dimension.
        bins_slice : int, list[int], list[ndarray]
            Specifies the bins used for slicing in `axis_slice`. The bin range
            is determined by the min/max point in `X`.
        axis_view, axis_slice : 2-tuple of int
            The axis to view (plot) and to slice.
        pad : int, float, list
            Fractional padding added to the start/stop indices for the bins in the sliced
            dimensions.
        debug : bool
            Whether to print debugging messages.
        **kws
            Key word arguments pass to `visualization.cloud.plot2d`
        """

        warnings.warn(
            "This is not currently working. For the time being, it is recommended to generate a 4D histogram and call `plot_image`."
        )

        # Setup
        # -----------------------------------------------------------------------
        if X.shape[1] < 4:
            raise ValueError(f"X.shape[1] = {X.shape[1]} < 4")
        self.axis_view = axis_view
        self.axis_slice = axis_slice

        edges_slice = psdist.cloud.histogram_bin_edges(X[:, axis_slice], bins_slice)

        # Select slice indices.
        if type(pad) in [float, int]:
            pad = len(axis_slice) * [pad]
        ind_slice = []
        for i, (steps, _pad) in enumerate(zip([self.nrows, self.ncols], pad)):
            start = int(_pad * len(edges_slice[i]))
            stop = len(edges_slice[i]) - 1 - start
            if (stop - start) < steps:
                raise ValueError(f"Too many slices requested.")
            ind_slice.append(np.linspace(start, stop, steps + 1).astype(int))
        ind_slice = tuple(ind_slice)
        self.ind_slice = ind_slice

        # Slice the bin indices.
        edges_slice = [e[ind] for e, ind in zip(edges_slice, ind_slice)]

        # Add dimension labels to the figure.
        if labels is not None:
            if self.annotate and labels is not None:
                self._annotate(
                    labels=[labels[k] for k in axis_view + axis_slice],
                    slice_label_height=self.slice_label_height,
                    annotate_kws_view=self.annotate_kws_view,
                    annotate_kws_slice=self.annotate_kws_slice,
                )

        # Plotting
        # -----------------------------------------------------------------------
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[self.nrows - 1 - i, j]
                _X = psdist.cloud.slice_planar(
                    X,
                    axis=axis_slice,
                    center=[
                        0.5 * (edges_slice[0][j] + edges_slice[0][j + 1]),
                        0.5 * (edges_slice[1][i] + edges_slice[1][i + 1]),
                    ],
                    width=[
                        np.abs(edges_slice[0][j] - edges_slice[1][j + 1]),
                        np.abs(edges_slice[1][i] - edges_slice[1][i + 1]),
                    ],
                )
                vis_cloud.plot2d(_X[:, axis_slice], ax=ax, **kws)
