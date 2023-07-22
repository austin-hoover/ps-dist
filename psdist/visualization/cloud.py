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


def auto_limits(X, sigma=None, pad=0.0, zero_center=False, share=None):
    """Determine axis limits from coordinate array.

    Parameters
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
    share : tuple[int] or list[tuple[int]]
        Limits are shared betweent the dimensions in each set. For example,
        if `share=(0, 1)`, axis 0 and 1 will share limits. Or if
        `share=[(0, 1), (4, 5)]` axis 0/1 will share limits, and axis 4/5
        will share limits.

    Returns
    -------
    limits : list[tuple]
        The limits [(xmin, xmax), (ymin, ymax), ...].
    """
    if X.ndim == 1:
        X = X[:, None]
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
    limits = [(_min, _max) for _min, _max in zip(mins, maxs)]
    if share:
        if np.ndim(share[0]) == 0:
            share = [share]
        for axis in share:
            _min = min([limits[k][0] for k in axis])
            _max = max([limits[k][1] for k in axis])
            for k in axis:
                limits[k] = (_min, _max)
    if zero_center:
        limits = psdist.visualization.center_limits(limits)
    if len(limits) == 1:
        limits = limits[0]
    return limits


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
    return psdist.visualization.rms_ellipse(
        Sigma, center, level=level, ax=ax, **ellipse_kws
    )


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
    """Two-dimensional density plot.

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
        Key word arguments passed to `visualization.plot_profile`.
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
        Key word argument passed to `visualization.plot_profile`.
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
        grid.set_labels(labels)
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
    share_limits=1,
    default_axis=(0, 1),
    slice_type="int",
    plot_res=64,
    slice_res=16,
    dims=None,
    units=None,
    options=None,
    autolim_kws=None,
    fig_kws=None,
    **plot_kws,
):
    """Two-dimensional partial projection of one or more clouds (or series of clouds)
    with interactive slicing.

    Parameters
    ----------
    data : ndarray, shape (n, d) or list[ndarray] or list[list[ndarray]]
        - Coordinates of n points in d-dimensional space.
        - List of L clouds: generates widget to select the frame to plot.
        - K lists of L clouds: generates K-column figure with widget to select one
          of the L frames.
          Example: Compare the evolution of K=3 bunches at L=6 frames.
    limits : list[(min, max)]
        Limits along each axis.
    share_limits : 0, 1, 2
        Whether to share axis limits across frames. If 0, don't share. If 1, share
        between each subplot for each figure (frame). If 2, share for all
        figures/subplots.
    default_axis : (int, int)
        Default view axis.
    slice_type : {"int", "range"}
        Whether to slice one index along the axis or a range of indices.
    plot_res, slice_res : int
        Default grid resolution for plotting/slicing. These can be updated using
        the interactive widgets.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    options : dict
        Determines the widgets to be displayed. Options are:
        - "auto_plot_res": Option to set bins="auto" in histogram.
        - "discrete": Option for discrete colormap norm. (Default: False).
        - "ellipse": Option to plot rms ellipse. (Default: False)
        - "log": Option for logarithmic colormap scaling. (Default: True)
        - "mask": Option to include a small offset so that there are no zero bins. (Default: False)
        - "normalize": Option to normalize x-px, y-py, z-pz to unit covariance matrix. (Default: False)
        - "profiles": Option to plot profiles (line-outs) on bottom and left spines.
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
        data = [[data[k]] for k in range(len(data))]

    n_rows = len(data)
    n_cols = len(data[0])
    n_dims = data[0][0].shape[1]

    for i in range(n_rows):
        if len(data[i]) != n_cols:
            raise ValueError("lists must have same length")

    for i in range(n_rows):
        for j in range(n_cols):
            if data[i][j].shape[1] != n_dims:
                raise ValueError("data must have the same number of dimensions.")

    if fig_kws is None:
        fig_kws = dict()

    plot_kws.setdefault("kind", "hist")
    plot_kws.setdefault(
        "rms_ellipse_kws",
        {
            "level": [1.0, 2.0, 3.0, 4.0, 5.0],
            "color": "white",
            "alpha": 0.2,
            "lw": 0.4,
        },
    )

    # Compute limits [(xmin, xmax), ...] for each data[i][j].
    if autolim_kws is None:
        autolim_kws = dict()
    if limits is None:
        limits_list = np.zeros((n_rows, n_cols, n_dims, 2))
        for i in range(n_rows):
            for j in range(n_cols):
                limits_list[i, j, :, :] = auto_limits(data[i][j], **autolim_kws)
        if share_limits == 1:
            for j in range(n_cols):
                limits = psdist.visualization.combine_limits(limits_list[:, j, :, :])
                for i in range(n_rows):
                    limits_list[i, j] = limits
        elif share_limits == 2:
            limits = psdist.visualization.combine_limits(
                limits_list.reshape((n_rows * n_cols, n_dims, 2))
            )
            for i in range(n_rows):
                for j in range(n_cols):
                    limits_list[i, j] = limits
    else:
        limits_list = [[limits for _ in range(n_cols)] for _ in range(n_rows)]

    # Set axis labels.
    if dims is None:
        dims = [f"x{i + 1}" for i in range(n_dims)]
    if units is None:
        units = n_dims * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)

    # Widgets
    _widgets = dict()
    _widgets["dim1"] = widgets.Dropdown(
        options=dims, index=default_axis[0], description="dim 1"
    )
    _widgets["dim2"] = widgets.Dropdown(
        options=dims, index=default_axis[1], description="dim 2"
    )
    _widgets["frame"] = widgets.BoundedIntText(
        min=0, max=(n_cols - 1), description="frame"
    )
    _widgets["slice_res"] = widgets.BoundedIntText(
        value=slice_res,
        min=2,
        max=200,
        step=1,
        description="slice_res",
    )
    _widgets["plot_res"] = widgets.BoundedIntText(
        value=plot_res,
        min=2,
        max=350,
        step=1,
        description="plot_res",
    )
    _widgets["auto_plot_res"] = widgets.Checkbox(
        description="auto_plot_res", value=False
    )
    _widgets["discrete"] = widgets.Checkbox(description="discrete", value=False)
    _widgets["ellipse"] = widgets.Checkbox(description="ellipse", value=False)
    _widgets["log"] = widgets.Checkbox(description="log", value=False)
    _widgets["mask"] = widgets.Checkbox(description="mask", value=True)
    _widgets["normalize"] = widgets.Checkbox(description="normalize", value=False)
    _widgets["profiles"] = widgets.Checkbox(description="profiles", value=False)

    # Sliders and checkboxes for slicing:
    _widgets["sliders"] = []
    _widgets["checks"] = []
    for k in range(n_dims):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=(_widgets["slice_res"].value - 1),
                value=int(_widgets["slice_res"].value / 2),
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                min=0,
                max=(_widgets["slice_res"].value - 1),
                value=(0, _widgets["slice_res"].value - 1),
                description=dims[k],
                continuous_update=True,
            )
        else:
            raise ValueError("Invalid `slice_type`.")
        slider.layout.display = "none"
        _widgets["sliders"].append(slider)
        _widgets["checks"].append(widgets.Checkbox(description=f"slice {dims[k]}"))

    def hide(button):
        """Hide inactive sliders."""
        for k in range(n_dims):
            # Hide elements for dimensions being plotted.
            valid = dims[k] not in [_widgets["dim1"].value, _widgets["dim2"].value]
            disp = None if valid else "none"
            for element in [_widgets["sliders"][k], _widgets["checks"][k]]:
                element.layout.display = disp
            # Uncheck boxes for dimensions being plotted.
            if not valid and _widgets["checks"][k].value:
                _widgets["checks"][k].value = False
            # Make sliders respond to check boxes.
            if not _widgets["checks"][k].value:
                _widgets["sliders"][k].layout.display = "none"
            _widgets["plot_res"].layout.display = (
                "none" if _widgets["auto_plot_res"].value else None
            )

    # Make slider visiblity depend on checkmarks.
    for element in (
        _widgets["dim1"],
        _widgets["dim2"],
        *_widgets["checks"],
        _widgets["auto_plot_res"],
    ):
        element.observe(hide, names="value")

    # Initial hide
    for k in range(n_dims):
        if k in default_axis:
            _widgets["checks"][k].layout.display = "none"
            _widgets["sliders"][k].layout.display = "none"
    if n_cols == 1:
        _widgets["frame"].layout.display = "none"

    # Set default options.
    if options is None:
        options = dict()
    options.setdefault("auto_plot_res", False)
    options.setdefault("discrete", False)
    options.setdefault("ellipse", False)
    options.setdefault("log", True)
    options.setdefault("mask", False)
    options.setdefault("normalize", False)
    options.setdefault("profiles", False)

    # Show/hide widgets based on `options`.
    for name, setting in options.items():
        _widgets[name].layout.display = None if setting else "none"

    def update(**kws):
        # Collect key word arguments.
        frame = kws["frame"]
        dim1 = kws["dim1"]
        dim2 = kws["dim2"]
        slice_res = kws["slice_res"]
        plot_res = kws["plot_res"]
        auto_plot_res = kws["auto_plot_res"]

        # Update the slider ranges/values based on slice_res.
        for slider in _widgets["sliders"]:
            slider.max = slice_res - 1

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

        # Collect data.
        _data = [data[index][frame] for index in range(n_rows)]
        _limits_list = limits_list[:, frame, :, :]

        # Normalize coordinates.
        if kws["normalize"]:
            _data = [
                psdist.cloud.norm_xxp_yyp_zzp(_X, scale_emittance=True) for _X in _data
            ]
            _limits_list = [auto_limits(_X, **autolim_kws) for _X in _data]
            if share_limits > 0:
                _limits_list = [
                    psdist.visualization.combine_limits(_limits_list)
                    for _ in range(len(_data))
                ]

        # Slice.
        axis_view = [dims.index(dim) for dim in (dim1, dim2)]
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        if axis_slice:
            for index in range(n_rows):
                slice_limits = []
                for k in axis_slice:
                    (imin, imax) = ind[k]
                    (xmin, xmax) = _limits_list[index][k]
                    edges = np.linspace(xmin, xmax, slice_res + 1)
                    if imax > len(edges) - 1:
                        print(f"{dims[k]} out of range.")
                        return
                    slice_limits.append((edges[imin], edges[imax]))
                _data[index] = psdist.cloud.slice_planar(
                    _data[index], axis=axis_slice, limits=slice_limits
                )

        # Handle empty slice (do nothing).
        for _X in _data:
            if _X.shape[0] == 0:
                return

        # Update plotting key word arguments.
        if plot_kws["kind"] != "scatter":
            plot_kws["bins"] = "auto" if auto_plot_res else plot_res
            plot_kws["discrete"] = kws["discrete"]
            plot_kws["mask"] = kws["mask"]
            plot_kws["norm"] = "log" if kws["log"] else None
            plot_kws["profx"] = kws["profiles"]
            plot_kws["profy"] = kws["profiles"]
            plot_kws["rms_ellipse"] = kws["ellipse"]

            # Add a small offset to the image if we are not masking and are using a logarithmic colormap.
            if plot_kws["norm"] == "log" and not plot_kws["mask"]:
                plot_kws.setdefault("offset", 1.0)

            # Temporary bug fix: If we check and then uncheck "log", and
            # the colorbar has minor ticks, the tick label formatter will
            # remain in "log" mode forever after.
            if "colorbar_kw" in plot_kws:
                if "tickminor" in plot_kws["colorbar_kw"] and not kws["log"]:
                    plot_kws["colorbar_kw"]["formatter"] = None

        # Create figure.
        fig, axs = pplt.subplots(
            ncols=n_rows,
            sharex=(share_limits and n_rows),
            sharey=(share_limits and n_rows),
            **fig_kws,
        )
        for index, ax in enumerate(axs):
            limits = [_limits_list[index][k] for k in axis_view]
            if plot_kws["kind"] != "scatter":
                plot_kws["limits"] = limits
            plot2d(_data[index][:, axis_view], ax=ax, **plot_kws)
            ax.format(xlim=limits[0], ylim=limits[1])
        labels = dims if _widgets["normalize"].value else dims_units
        axs.format(
            xlabel=labels[axis_view[0]],
            ylabel=labels[axis_view[1]],
        )
        plt.show()

    # Pass key word arguments to `ipywidgets.interactive`.
    kws = dict()
    for key in [
        "frame",
        "dim1",
        "dim2",
        "slice_res",
        "plot_res",
        "auto_plot_res",
        "discrete",
        "ellipse",
        "log",
        "mask",
        "normalize",
        "profiles",
    ]:
        kws[key] = _widgets[key]
    for i, check in enumerate(_widgets["checks"], start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(_widgets["sliders"], start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)


def proj1d_interactive_slice(
    data=None,
    limits=None,
    share_limits=False,
    default_axis=0,
    slice_type="int",
    plot_res=64,
    slice_res=16,
    dims=None,
    units=None,
    colors=None,
    cycle=None,
    options=None,
    labels=None,
    legend=False,
    update_limits_on_slice=False,
    legend_kws=None,
    autolim_kws=None,
    fig_kws=None,
    **plot_kws,
):
    """One-dimensional partial projection of one or more clouds (or series of clouds)
    with interactive slicing.

    Profiles are scaled to unit area by default.

    Parameters
    ----------
    data : ndarray, shape (n, d) or list[ndarray] or list[list[ndarray]]
        - Coordinates of n points in d-dimensional space.
        - List of L clouds: generates widget to select the frame to plot.
        - K lists of L clouds: generates K-column figure with widget to select one
          of the L frames.
          Example: Compare the evolution of K=3 bunches at L=6 frames.
    limits : list[(min, max)]
        Limits along each axis.
    share_limits : bool
        Whether to share axis limits across frames.
    default_axis : int
        Default view axis.
    slice_type : {"int", "range"}
        Whether to slice one index along the axis or a range of indices.
    plot_res, slice_res : int
        Default grid resolution for plotting/slicing. These can be updated using
        the interactive widgets.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    colors : list
        List of colors, otherwise use default color cycle.
    cycle : str
        If the name of a cycle is passed, override `colors`.
    options : dict
        Determines the widgets to be displayed. Options are:
        - "alpha": Option to scale plot opacity (alpha). (Default: False)
        - "auto_plot_res": Option to set bins="auto" in histogram. (Default: False)
        - "log": Option for logarithmic y axis scale. (Default: True)
        - "normalize": Option to normalize x-px, y-py, z-pz to unit covariance matrix. (Default: False)
        - "scale": Option to scale to unit max or unit area. (Default: False)
    labels : list[str]
        Labels for legend.
    legend : bool
        Whether to include legend.
    update_limits_on_slice : bool
        Recompute limits when the distribution is sliced. Otherwise, keep the limits
        from the full distribution.
    legend_kws : bool
        Key word arguments for legend.
    autolim_kws : dict
        Key word arguments passed to `auto_limits`.
    fig_kws : dict
        Key word arguments passed to `proplot.subplots`.
    **plot_kws
        Key word arguments passed to `plot_profile`.
    """
    if type(data) is not list:
        data = [data]
    if type(data[0]) is not list:
        data = [[data[k]] for k in range(len(data))]

    n_rows = len(data)
    n_cols = len(data[0])
    n_dims = data[0][0].shape[1]

    for i in range(n_rows):
        if len(data[i]) != n_cols:
            raise ValueError("lists must have same length")

    for i in range(n_rows):
        for j in range(n_cols):
            if data[i][j].shape[1] != n_dims:
                raise ValueError("data must have the same number of dimensions.")

    if fig_kws is None:
        fig_kws = dict()

    # Plot color cycle.
    if colors is None:
        if n_rows == 1:
            colors = ["black"]
        if cycle is None:
            colors = pplt.Cycle(pplt.rc["cycle"]).by_key()["color"]
        else:
            colors = pplt.Cycle(cycle).by_key()["color"]

    # Legend
    if legend_kws is None:
        legend_kws = dict()
        legend_kws.setdefault("loc", "right")
        legend_kws.setdefault("framealpha", 0.0)
        legend_kws.setdefault("ncols", 1)
    if labels is None:
        labels = [f"prof_{i}" for i in range(n_rows)]

    # Compute limits [(xmin, xmax), ...] for each frame.
    if autolim_kws is None:
        autolim_kws = dict()
    if limits is None:
        limits_list = np.zeros((n_cols, n_dims, 2))
        for j in range(n_cols):
            limits_list[j] = psdist.visualization.combine_limits(
                [auto_limits(data[i][j], **autolim_kws) for i in range(n_rows)]
            )
        if share_limits:
            limits = psdist.visualization.combine_limits(limits_list)
            for j in range(n_cols):
                limits_list[j] = limits
    else:
        limits_list = [[limits for _ in range(n_cols)] for _ in range(n_rows)]

    # Set axis labels.
    if dims is None:
        dims = [f"x{i + 1}" for i in range(n_dims)]
    if units is None:
        units = n_dims * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)

    # Widgets
    _widgets = dict()
    _widgets["dim"] = widgets.Dropdown(
        options=dims, index=default_axis, description="dim"
    )
    _widgets["frame"] = widgets.BoundedIntText(
        min=0, max=(n_cols - 1), description="frame"
    )
    _widgets["slice_res"] = widgets.BoundedIntText(
        value=slice_res,
        min=2,
        max=200,
        step=1,
        description="slice_res",
    )
    _widgets["plot_res"] = widgets.BoundedIntText(
        value=plot_res,
        min=2,
        max=350,
        step=1,
        description="plot_res",
    )
    _widgets["alpha"] = widgets.FloatSlider(
        min=0.0, max=1.0, value=1.0, description="alpha"
    )
    _widgets["auto_plot_res"] = widgets.Checkbox(
        description="auto_plot_res", value=False
    )
    _widgets["kind"] = widgets.Dropdown(
        description="kind",
        options=["step", "stepfilled", "line", "linefilled"],
        value="step",
    )
    _widgets["log"] = widgets.Checkbox(description="log", value=False)
    _widgets["normalize"] = widgets.Checkbox(description="normalize", value=False)
    _widgets["scale"] = widgets.Dropdown(
        options=["none", "density", "max"], description="scale", value="density"
    )

    # Sliders and checkboxes for slicing:
    _widgets["sliders"] = []
    _widgets["checks"] = []
    for k in range(n_dims):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=(_widgets["slice_res"].value - 1),
                value=int(_widgets["slice_res"].value / 2),
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                min=0,
                max=(_widgets["slice_res"].value - 1),
                value=(0, _widgets["slice_res"].value - 1),
                description=dims[k],
                continuous_update=True,
            )
        else:
            raise ValueError("Invalid `slice_type`.")
        slider.layout.display = "none"
        _widgets["sliders"].append(slider)
        _widgets["checks"].append(widgets.Checkbox(description=f"slice {dims[k]}"))

    def hide(button):
        """Hide inactive sliders."""
        for k in range(n_dims):
            # Hide elements for dimensions being plotted.
            valid = dims[k] != _widgets["dim"].value
            disp = None if valid else "none"
            for element in [_widgets["sliders"][k], _widgets["checks"][k]]:
                element.layout.display = disp
            # Uncheck boxes for dimensions being plotted.
            if not valid and _widgets["checks"][k].value:
                _widgets["checks"][k].value = False
            # Make sliders respond to check boxes.
            if not _widgets["checks"][k].value:
                _widgets["sliders"][k].layout.display = "none"
            _widgets["plot_res"].layout.display = (
                "none" if _widgets["auto_plot_res"].value else None
            )

    # Make slider visiblity depend on checkmarks.
    for element in [_widgets["dim"], *_widgets["checks"], _widgets["auto_plot_res"]]:
        element.observe(hide, names="value")

    # Initial hide
    for k in range(n_dims):
        if k == default_axis:
            _widgets["checks"][k].layout.display = "none"
            _widgets["sliders"][k].layout.display = "none"
    if n_cols == 1:
        _widgets["frame"].layout.display = "none"

    # Set default options.
    if options is None:
        options = dict()
    options.setdefault("alpha", False)
    options.setdefault("auto_plot_res", False)
    options.setdefault("kind", False)
    options.setdefault("scale", False)
    options.setdefault("log", True)
    options.setdefault("normalize", False)

    # Show/hide widgets based on `options`.
    for name, setting in options.items():
        _widgets[name].layout.display = None if setting else "none"

    def update(**kws):
        # Collect key word arguments.
        frame = kws["frame"]
        dim_view = kws["dim"]

        # Update the slider ranges/values based on slice_res.
        for slider in _widgets["sliders"]:
            slider.max = kws["slice_res"] - 1

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
            if check and dim == dim_view:
                return

        # Collect data.
        _data = [data[index][frame] for index in range(n_rows)]
        limits = limits_list[frame, :, :]

        # Normalize coordinates.
        if kws["normalize"]:
            _data = [
                psdist.cloud.norm_xxp_yyp_zzp(_X, scale_emittance=True) for _X in _data
            ]
            limits = psdist.visualization.combine_limits(
                [auto_limits(_X, **autolim_kws) for _X in _data]
            )

        # Slice.
        axis_view = dims.index(dim_view)
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        if axis_slice:
            for index in range(n_rows):
                slice_limits = []
                for k in axis_slice:
                    (imin, imax) = ind[k]
                    (xmin, xmax) = limits[k]
                    edges = np.linspace(xmin, xmax, kws["slice_res"] + 1)
                    if imax > len(edges) - 1:
                        print(f"{dims[k]} out of range.")
                        return
                    slice_limits.append((edges[imin], edges[imax]))
                _data[index] = psdist.cloud.slice_planar(
                    _data[index], axis=axis_slice, limits=slice_limits
                )

        # Handle empty slice (do nothing).
        for _X in _data:
            if _X.shape[0] == 0:
                return

        # Recompute limits from sliced data.
        if update_limits_on_slice:
            limits = psdist.visualization.combine_limits(
                [auto_limits(_X, **autolim_kws) for _X in _data]
            )

        # Create figure.
        bins = "auto" if kws["auto_plot_res"] else kws["plot_res"]

        fig, ax = pplt.subplots(**fig_kws)
        for index in range(len(_data)):
            # Bin data
            points = _data[index][:, axis_view]
            bin_edges = np.histogram_bin_edges(points, bins, limits[axis_view])
            hist, _ = np.histogram(points, bins=bin_edges)

            # Plot
            plot_kws["color"] = colors[index]
            plot_kws["label"] = labels[index]
            plot_kws["alpha"] = kws["alpha"]
            plot_kws["kind"] = kws["kind"]
            plot_kws["scale"] = kws["scale"]
            psdist.visualization.plot_profile(
                profile=hist, edges=bin_edges, ax=ax, **plot_kws
            )

        if kws["log"]:
            ax.format(yscale="log", yformatter="log")
        ax.format(
            xlim=limits[axis_view],
            xlabel=dims[axis_view]
            if _widgets["normalize"].value
            else dims_units[axis_view],
        )
        if legend and (labels is not None):
            ax.legend(**legend_kws)
        plt.show()

    # Pass key word arguments to `ipywidgets.interactive`.
    kws = dict()
    for key in [
        "frame",
        "dim",
        "slice_res",
        "plot_res",
        "alpha",
        "auto_plot_res",
        "kind",
        "log",
        "normalize",
        "scale",
    ]:
        kws[key] = _widgets[key]
    for i, check in enumerate(_widgets["checks"], start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(_widgets["sliders"], start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)
