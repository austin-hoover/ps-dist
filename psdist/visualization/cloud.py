"""Plotting routines for point data."""
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt

import psdist.cloud
import psdist.image
import psdist.visualization.image as vis_image
import psdist.visualization.visualization as vis


def auto_limits(X, sigma=None, pad=0.0, zero_center=False, share_xy=False):
    """Determine axis limits from coordinate array.

    Parametersv
    ----------
    X : ndarray, shape (n, d)
        Coordinate array for n points in d-dimensional space.
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
    X : ndarray, shape (n, 2)
        Coordinate array for n points in 2-dimensional space.
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
    X : ndarray, shape (n, d)
        Coordinate array for n points in d-dimensional space.
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
    kws.setdefault("ec", "None")
    kws.setdefault("s", 2.0)
    _X = X
    if samples:
        _X = psdist.cloud.downsample(X, samples)
    return ax.scatter(_X[:, 0], _X[:, 1], **kws)


def hist(X, ax=None, bins="auto", limits=None, **kws):
    """Convenience function for 2D histogram with auto-binning.

    For more options, I recommend seaborn.histplot.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
        Coordinate array for n points in 2-dimensional space.
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
    X : ndarray, shape (n, 2)
        Coordinate array for n points in 2-dimensional space.
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
    X : ndarray, shape (n, d)
        Coordinate array for n points in d-dimensional space.
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

    This is a convenience function; see `psdist.visualization.grid.JointGrid`.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
        Coordinates of n points in 2-dimensional space.
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
    psdist.visualization.grid.JointGrid
    """
    from psdist.visualization.grid import JointGrid
    if grid_kws is None:
        grid_kws = dict()
    grid = JointGrid(**grid_kws)
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

    This is a convenience function; see `psdist.visualization.grid.CornerGrid`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
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
    psdist.visualization.grid.CornerGrid
        The `CornerGrid` on which the plot was drawn.
    """
    from psdist.visualization.grid import CornerGrid
    if grid_kws is None:
        grid_kws = dict()
    grid = CornerGrid(d=X.shape[1], **grid_kws)
    if labels is not None:
        cgrid.set_labels(labels)
    grid.plot_cloud(
        X,
        limits=limits,
        autolim_kws=autolim_kws,
        prof_edge_only=prof_edge_only,
        update_limits=update_limits,
        diag_kws=diag_kws,
        **kws,
    )
    return grid


def proj2d_interactive_slice(
    data=None,
    limits=None,
    default_ind=(0, 1),
    slice_type="int",
    plot_res=64,
    slice_res=16,
    dims=None,
    units=None,
    autolim_kws=None,
    fig_kws=None,
    **plot_kws,
):
    """2D partial projection of one or more clouds with interactive slicing.

    Parameters
    ----------
    data : ndarray, shape (n, d) or list[ndarray] or list[list[ndarray]]
        - Coordinates of n points in d-dimensional space. 
        - List of K clouds: generates K-column figure.
        - K lists of L clouds: generates K-column figure with widget to select one
          of the L frames. Example: Compare the evolution of K=3 bunches at L=6 
          frames (times).
    limits : list[(min, max)]
        Limits along each axis.
    default_ind : (i, j)
        Default view axis.
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    plot_res, slice_res : int
        Default grid resolution for plotting/slicing. These can be updated using
        the interactive widgets.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    autolim_kws : dict
        Key word arguments passed to `auto_limits`.
    fig_kws : dict
        Key word arguments passed to `proplot.subplots`.
    **plot_kws
        Key word arguments passed to `plot2d`.
    """
    if type(data) is not list:
        data = [data]
    if type(data[0]) is not list:
        data = [data]

    n_clouds = len(data)
    n_frames = len(data[0])
    n_dims = data[0][0].shape[1]
    
    for i in range(n_clouds):
        if len(data[i]) != n_frames:
            raise ValueError("lists must have same length")
            
    for i in range(n_clouds):
        for j in range(n_frames):
            if data[i][j].shape[1] != n_dims:
                raise ValueError("data must have the same number of dimensions.")

    if fig_kws is None:
        fig_kws = dict()
    plot_kws.setdefault("kind", "hist")    

    if limits is None:
        # Compute min/max across all clouds and frames.
        if autolim_kws is None:
            autolim_kws = dict()
        limits_list = [
            auto_limits(data[i][j], **autolim_kws)
            for i in range(n_clouds)
            for j in range(n_frames)
        ]
        limits_list = np.array(limits_list)  # cloud, frame, dim, min/max
        mins = np.min(limits_list[:, :, 0], axis=0)    
        maxs = np.max(limits_list[:, :, 1], axis=0)
        limits = [(mins[i], maxs[i]) for i in range(n_dims)]
    if dims is None:
        dims = [f"x{i + 1}" for i in range(n_dims)]
    if units is None:
        units = n_dims * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)

    # Widgets
    dim1 = widgets.Dropdown(options=dims, index=default_ind[0], description="dim 1")
    dim2 = widgets.Dropdown(options=dims, index=default_ind[1], description="dim 2")
    frame = widgets.BoundedIntText(min=0, max=(n_frames - 1), description="frame")
    n_bins = widgets.BoundedIntText(
        value=slice_res,
        min=2,
        max=200,
        step=1,
        description="slice res",
    )    
    n_bins_plot = widgets.BoundedIntText(
        value=plot_res,
        min=2,
        max=350,
        step=1,
        description="plot res",
    )    
    autobin = widgets.Checkbox(description="auto plot res", value=False)
    log = widgets.Checkbox(description="log", value=False)
    sliders, checks = [], []
    for k in range(n_dims):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=(n_bins.value - 1),
                value=int(n_bins.value / 2),
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                min=0,
                max=(n_bins.value - 1),
                value=(0, n_bins.value - 1),
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
        for k in range(n_dims):
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
            n_bins_plot.layout.display = "none" if autobin.value else None

    # Make slider visiblity depend on checkmarks.
    for element in (dim1, dim2, *checks, autobin):
        element.observe(hide, names="value")

    # Initial hide
    for k in range(n_dims):
        if k in default_ind:
            checks[k].layout.display = "none"
            sliders[k].layout.display = "none"
    if n_frames == 1:
        frame.layout.display = "none"

    def update(**kws):
        frame = kws["frame"]
        dim1 = kws["dim1"]
        dim2 = kws["dim2"]
        n_bins = kws["n_bins"]
        n_bins_plot = kws["n_bins_plot"]
        autobin = kws["autobin"]

        # Update the slider ranges/values based on n_bins.
        for slider in sliders:
            slider.max = n_bins - 1

        # Collect slice indices.
        ind, checks = [], []
        for i in range(1, n_dims + 1):
            if f"check{i}" in kws:
                checks.append(kws[f"check{i}"])
            if f"slider{i}" in kws:
                _ind = kws[f"slider{i}"]
                if type(_ind) is int:
                    _ind = (_ind, _ind + 1)
                ind.append(_ind)

        # Exit if input is invalid.
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        if dim1 == dim2:
            return     

        # Slice the distributions.
        axis_view = [dims.index(dim) for dim in (dim1, dim2)]
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        edges = [np.linspace(limits[i][0], limits[i][1], n_bins + 1) for i in range(n_dims)]
        _data = []
        if not axis_slice:
            _data = [data[i][frame] for i in range(n_clouds)]
        else:
            slice_limits = []
            for k in axis_slice:
                imin, imax = ind[k]
                if imax > len(edges[k]) - 1:
                    print(f"{dims[k]} out of range.")
                    return
                slice_limits.append((edges[k][imin], edges[k][imax]))
            _data = [
                psdist.cloud.slice_planar(data[i][frame], axis=axis_slice, limits=slice_limits)
                for i in range(n_clouds)
            ]
        for _X in _data:
            if _X.shape[0] == 0:
                return
                
        # Update plotting key word arguments.
        if plot_kws["kind"] != "scatter":
            plot_kws["bins"] = "auto" if autobin else n_bins_plot
            plot_kws["limits"] = [limits[axis_view[0]], limits[axis_view[1]]]
            plot_kws["norm"] = "log" if kws["log"] else None
            
            # Temporary bug fix: If we check and then uncheck "log", and 
            # the colorbar has minor ticks, the tick label formatter will
            # remain in "log" mode forever after.
            if "colorbar_kw" in plot_kws:
                if "tickminor" in plot_kws["colorbar_kw"] and not kws["log"]:
                    plot_kws["colorbar_kw"]["formatter"] = None

        # Plot the selected points.
        fig, axs = pplt.subplots(ncols=n_clouds, **fig_kws)
        for ax, _X in zip(axs, _data):
            plot2d(_X[:, axis_view], ax=ax, **plot_kws)
        axs.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()

    # Pass key word arguments to `ipywidgets.interactive`.
    kws = dict()
    kws["log"] = log
    kws["autobin"] = autobin
    kws["n_bins"] = n_bins
    kws["n_bins_plot"] = n_bins_plot
    kws["frame"] = frame
    kws["dim1"] = dim1
    kws["dim2"] = dim2
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)
