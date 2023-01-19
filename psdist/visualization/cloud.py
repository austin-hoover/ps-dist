"""Plotting routines for discrete sets of points in 2n-dimensional phase space."""
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt

import psdist.cloud
import psdist.image
import psdist.visualization.visualization as vis
import psdist.visualization.image as vis_image


def auto_limits(X, sigma=None, pad=0.0, zero_center=False, share_xy=False):
    """Determine axis limits from coordinate array.

    Parametersv
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    sigma : float
        If a number is provided, it is used to set the limits relative to
        the standard deviation of the distribution.
    pad : float
        Fractional padding to apply to the limits.
    zero_center : bool
        Whether to center the limits on zero.
    share_xy : bool
        Whether to share x/y limits (and x'/y').

    Returns
    -------
    mins, maxs : list
        The new limits.
    """
    if sigma is None:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        widths = 2.0 * sigma * stds
        mins = means - 0.5 * widths
        maxs = means + 0.5 * widths
    deltas = 0.5 * np.abs(maxs - mins)
    padding = deltas * pad
    mins = mins - padding
    maxs = maxs + padding
    if zero_center:
        maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
        mins = -maxs
    if share_xy:
        widths = np.abs(mins - maxs)
        for (i, j) in [[0, 2], [1, 3]]:
            delta = 0.5 * (widths[i] - widths[j])
            if delta < 0.0:
                mins[i] -= abs(delta)
                maxs[i] += abs(delta)
            elif delta > 0.0:
                mins[j] -= abs(delta)
                maxs[j] += abs(delta)
    return [(_min, _max) for _min, _max in zip(mins, maxs)]


def plot_rms_ellipse(X, ax=None, level=1.0, center_at_mean=True, **ellipse_kws):
    """Compute and plot RMS ellipse from bunch coordinates.

    Parameters
    ----------
    X : ndarray, shape (k, 2)
        Coordinate array for k points in 2-dimensional space.
    ax : Axes
        The axis on which to plot.
    level : number of list of numbers
        If a number, plot the rms ellipse inflated by the number. If a list
        of numbers, repeat for each number.
    center_at_mean : bool
        Whether to center the ellipse at the image centroid.
    """
    center = np.mean(X, axis=0)
    if center_at_mean:
        center = (0.0, 0.0)
    Sigma = np.cov(X.T)
    return vis.rms_ellipse(Sigma, center, level=level, ax=ax, **ellipse_kws)


def scatter(X, ax=None, samples=None, **kws):
    """Convenience function for 2D scatter plot.

    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    ax : Axes
        The axis on which to plot.
    samples : int
        Plot this many random samples.
    **kws
        Key word arguments passed to `ax.scatter`.
    """
    if "color" in kws:
        kws["c"] = kws.pop("color")
    for kw in ["size", "ms"]:
        if kw in kws:
            kws["s"] = kws.pop(kw)
    kws.setdefault("c", "black")
    kws.setdefault("s", 1.0)
    _X = X
    if samples:
        _X = psdist.cloud.downsample(X, samples)
    return ax.scatter(_X[:, 0], _X[:, 1], **kws)


def hist(X, ax=None, bins="auto", limits=None, **kws):
    """Convenience function for 2D histogram with auto-binning.

    For more options, I recommend seaborn.histplot.

    Parameters
    ----------
    X : ndarray, shape (k, 2)
        Coordinate array for k points in 2-dimensional space.
    ax : Axes
        The axis on which to plot.
    limits, bins : see `psdist.bunch.histogram`.
    **kws
        Key word arguments passed to `plotting.image`.
    """
    f, coords = psdist.cloud.histogram(X, bins=bins, limits=limits, centers=True)
    return vis_image.plot2d(f, coords=coords, ax=ax, **kws)


def kde(X, ax=None, coords=None, res=100, kde_kws=None, **kws):
    """Plot kernel density estimation (KDE).

    Parameters
    ----------
    X : ndarray, shape (k, 2)
        Coordinate array for k points in 2-dimensional space.
    ax : Axes
        The axis on which to plot.
    coords : [xcoords, ycoords]
        Coordinates along each axis of a two-dimensional regular grid on which to
        evaluate the density.
    res : int
        If coords is not provided, determines the evaluation grid resolution.
    kde_kws : dict
        Key word arguments passed to `psdist.bunch.kde`.
    **kws
        Key word arguments passed to `psdist.visualization.image.plot2`.
    """
    if kde_kws is None:
        kde_kws = dict()
    if coords is None:
        lb = np.min(X, axis=0)
        ub = np.max(X, axis=0)
        coords = [np.linspace(l, u, res) for l, u in zip(lb, ub)]
    estimator = psdist.cloud.gaussian_kde(X, **kde_kws)
    density = estimator.evaluate(psdist.image.get_grid_coords(*coords).T)
    density = np.reshape(density, [len(c) for c in coords])
    return vis_image.plot2d(density, coords=coords, ax=ax, **kws)


def plot2d(X, kind="hist", rms_ellipse=False, rms_ellipse_kws=None, ax=None, **kws):
    """Plot

    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    kind : {'hist', 'contour', 'contourf', 'scatter', 'kde'}
        The kind of plot.
    rms_ellipse : bool
        Whether to plot the RMS ellipse.
    rms_ellipse_kws : dict
        Key word arguments passed to `visualization.cloud.plot_rms_ellipse`.
    ax : Axes
        The axis on which to plot.
    **kws
        Key word arguments passed to `visualization.cloud.plot2d`.
    """
    if kind == "hist":
        kws.setdefault("mask", True)
    func = None
    if kind in ["hist", "contour", "contourf"]:
        func = hist
        if kind in ["contour", "contourf"]:
            kws["kind"] = kind
    elif kind == "scatter":
        func = scatter
    elif kind == "kde":
        func = kde
    else:
        raise ValueError("Invalid plot kind.")
    _out = func(X, ax=ax, **kws)
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_rms_ellipse(X, ax=ax, **rms_ellipse_kws)
    return _out


def joint(X, grid_kws=None, marg_hist_kws=None, marg_kws=None, **kws):
    """Joint plot.
    
    This is a convenience function; see `psdist.visualization.JointGrid`.

    Parameters
    ----------
    X : ndarray, shape (k, 2)
        Coordinates of k points in 2-dimensional space.
    grid_kws : dict
        Key word arguments passed to `JointGrid`.
    marg_hist_kws : dict
        Key word arguments passed to `np.histogram` for 1D histograms.
    marg_kws : dict
        Key word arguments passed to `visualization.plot1d`.
    **kws
        Key word arguments passed to `visualization.image.plot2d.`
        
    Returns
    -------
    psdist.visualization.JointGrid
    """
    if grid_kws is None:
        grid_kws = dict()
    grid = vis.JointGrid(**grid_kws)
    grid.plot_cloud(X, marg_hist_kws=marg_hist_kws, marg_kws=marg_kws, **kws)
    return grid


def corner(
    X,
    grid_kws=None,
    limits=None,
    labels=None,
    autolim_kws=None,
    prof_edge_only=False,
    update_limits=True,
    diag_kws=None,
    **kws,
):
    """Corner plot (scatter plot matrix).

    This is a convenience function; see `psdist.visualization.CornerGrid`.

    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    limits : list[tuple], length n
        The (min, max) plot limits for each axis.
    labels : list[str], length n
        The axis labels.
    prof_edge_only : bool
        If plotting profiles on top of images (on off-diagonal subplots), whether
        to plot x profiles only in bottom row and y profiles only in left column.
    update_limits : bool
        Whether to extend the existing plot limits.
    diag_kws : dict
        Key word argument passed to `visualization.plot1d`.
    **kws
        Key word arguments pass to `visualization.cloud.plot2d`

    Returns
    -------
    psdist.visualization.CornerGrid
        The `CornerGrid` on which the plot was drawn.
    """
    if grid_kws is None:
        grid_kws = dict()
    cgrid = vis.CornerGrid(n=X.shape[1], **grid_kws)
    if labels is not None:
        cgrid.set_labels(labels)
    cgrid.plot_cloud(
        X,
        limits=limits,
        autolim_kws=autolim_kws,
        prof_edge_only=prof_edge_only,
        update_limits=update_limits,
        diag_kws=diag_kws,
        **kws,
    )
    return cgrid


def proj2d_interactive_slice(
    X,
    limits=None,
    nbins=30,
    default_ind=(0, 1),
    slice_type="int",
    dims=None,
    units=None,
    **plot_kws,
):
    """2D partial projection of bunch with interactive slicing.

    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    limits : list[(min, max)]
        Limits along each axis.
    nbins : int
        Default number of bins for slicing/viewing. Both can be changed with
        sliders.
    default_ind : (i, j)
        Default view axis.
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    **plot_kws
        Key word arguments passed to `plot2d`.
    """
    plot_kws.setdefault("kind", "hist")

    n = X.shape[1]
    if limits is None:
        limits = [(np.min(X[:, i]), np.max(X[:, i])) for i in range(n)]
    if dims is None:
        dims = [f"x{i + 1}" for i in range(n)]
    if units is None:
        units = n * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)

    # Widgets
    nbins_default = nbins
    dim1 = widgets.Dropdown(options=dims, index=default_ind[0], description="dim 1")
    dim2 = widgets.Dropdown(options=dims, index=default_ind[1], description="dim 2")
    nbins = widgets.IntSlider(
        min=2, max=100, value=nbins_default, description="grid res"
    )
    nbins_plot = widgets.IntSlider(
        min=2, max=200, value=nbins_default, description="plot res"
    )
    autobin = widgets.Checkbox(description="auto plot res", value=False)
    log = widgets.Checkbox(description="log", value=False)
    sliders, checks = [], []
    for k in range(n):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=100,
                value=0,
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                value=(0, 100),
                min=0,
                max=100,
                description=dims[k],
                continuous_update=True,
            )
        else:
            raise ValueError("Invalid `slice_type`.")
        slider.layout.display = "none"
        sliders.append(slider)
        checks.append(widgets.Checkbox(description=f"slice {dims[k]}"))

    def hide(button):
        """Hide inactive sliders."""
        for k in range(n):
            # Hide elements for dimensions being plotted.
            valid = dims[k] not in (dim1.value, dim2.value)
            disp = None if valid else "none"
            for element in [sliders[k], checks[k]]:
                element.layout.display = disp
            # Uncheck boxes for dimensions being plotted.
            if not valid and checks[k].value:
                checks[k].value = False
            # Make sliders respond to check boxes.
            if not checks[k].value:
                sliders[k].layout.display = "none"
            nbins_plot.layout.display = "none" if autobin.value else None

    # Make slider visiblity depend on checkmarks.
    for element in (dim1, dim2, *checks, autobin):
        element.observe(hide, names="value")

    # Initial hide
    nbins_plot.layout.display = "none"
    for k in range(n):
        if k in default_ind:
            checks[k].layout.display = "none"
            sliders[k].layout.display = "none"

    def update(**kws):
        dim1 = kws["dim1"]
        dim2 = kws["dim2"]
        nbins = kws["nbins"]
        nbins_plot = kws["nbins_plot"]
        autobin = kws["autobin"]
        ind, checks = [], []
        for i in range(100):
            if f"check{i}" in kws:
                checks.append(kws[f"check{i}"])
            if f"slider{i}" in kws:
                _ind = kws[f"slider{i}"]
                if type(_ind) is int:
                    _ind = (_ind, _ind + 1)
                ind.append(_ind)

        # Exit if input does not make sense.
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        if dim1 == dim2:
            return

        # Slice the distribution
        axis_view = [dims.index(dim) for dim in (dim1, dim2)]
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        edges = [np.linspace(umin, umax, nbins + 1) for (umin, umax) in limits]
        if axis_slice:
            center, width = [], []
            for k in axis_slice:
                imin, imax = ind[k]
                if imax > len(edges[k]) - 1:
                    print(f"{dims[k]} out of range.")
                    return
                width.append(edges[k][imax] - edges[k][imin])
                center.append(0.5 * (edges[k][imax] + edges[k][imin]))
            _X = psdist.cloud.slice_planar(
                X, axis=axis_slice, center=center, width=width
            )
        else:
            _X = X[:, :]
        if _X.shape[0] == 0:
            return

        # Update plotting key word arguments.
        if plot_kws["kind"] != "scatter":
            plot_kws["bins"] = "auto" if autobin else nbins_plot
            plot_kws["limits"] = limits
            plot_kws["norm"] = "log" if kws["log"] else None

        # Plot the selected points.
        fig, ax = pplt.subplots()
        plot2d(_X[:, axis_view], ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()

    kws = dict()
    kws["log"] = log
    kws["autobin"] = autobin
    kws["dim1"] = dim1
    kws["dim2"] = dim2
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    kws["nbins"] = nbins
    kws["nbins_plot"] = nbins_plot
    return interactive(update, **kws)
