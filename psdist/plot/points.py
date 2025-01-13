"""Plotting routines for points."""

import numpy as np
import matplotlib.pyplot as plt
import ultraplot as uplt
from ipywidgets import interactive
from ipywidgets import widgets

from ..core import downsample as _downsample
from ..core import limits as get_limits
from ..core import combine_limits
from ..density import estimate_density as _estimate_density
from ..hist import Histogram
from ..hist import Histogram1D
from .. import core
from .core import plot_rms_ellipse_cov as _plot_rms_ellipse
from .hist import plot as _plot_hist
from .hist import plot_1d as _plot_hist_1d


def plot_hist_1d(
    points: np.ndarray,
    bins: int = 10,
    limits: np.ndarray = None,
    ax=None,
    **kws,
):
    """Plot one-dimensional histogram."""
    values, edges = np.histogram(points, bins=bins, range=limits)
    hist = Histogram1D(values, edges=edges)
    return _plot_hist_1d(hist, ax=ax, **kws)


def plot_density_estimate_1d(
    points: np.ndarray,
    bins: int = 10,
    limits: np.ndarray = None,
    method: str = "kde",
    method_kws: dict = None,
    ax=None,
    **kws,
):
    """Plot one-dimensional density estimate."""
    if method_kws is None:
        method_kws = {}

    if limits is None:
        limits = get_limits(points)

    edges = np.linspace(limits[0], limits[1], bins + 1)
    hist = Histogram1D(edges=edges)

    eval_points = hist.coords
    values = _estimate_density(points, eval_points, method, **method_kws)
    values = values.reshape(hist.shape)
    hist.values = values
    hist.normalize()
    return _plot_hist_1d(hist, ax=ax, **kws)


def plot_1d(points: np.ndarray, kde: bool = False, **kws):
    """Plot one-dimensional density."""
    plot_function = None
    if kde:
        plot_function = plot_density_estimate_1d
    else:
        plot_function = plot_hist_1d

    return plot_function(points, **kws)


def plot_rms_ellipse(
    points: np.ndarray,
    level: float | list[float] = 1.0,
    center_at_mean: bool = True,
    ax=None,
    **ellipse_kws,
):
    """Compute and plot RMS ellipse from bunch coordinates.

    Parameters
    ----------
    points : ndarray, shape (..., 2)
        Particle coordinates.
    level : number of list of numbers
        If a number, plot the rms ellipse inflated by the number. If a list of
        numbers, repeat for each number.
    center_at_mean : bool
        Whether to center the ellipse at the image centroid.
    """
    center = np.mean(points, axis=0)
    if not center_at_mean:
        center = (0.0, 0.0)
    return _plot_rms_ellipse(
        np.cov(points.T), center, level=level, ax=ax, **ellipse_kws
    )


def plot_scatter(points: np.ndarray, samples: int = None, ax=None, **kws):
    """Convenience function for 2D scatter plot.

    Parameters
    ----------
    points: np.ndarray, shape (..., n)
        Coordinate array for n points in d-dimensional space.
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

    if samples is not None:
        points = _downsample(points, samples)
    return ax.scatter(points[:, 0], points[:, 1], **kws)


def plot_hist(
    points: np.ndarray,
    bins: int = 50,
    limits: np.ndarray = None,
    ax=None,
    **kws,
):
    """Plot two-dimensional histogram.

    Parameters
    ----------
    points: np.ndarray, shape (..., n)
        Particle coordinates.
    limits, bins :
        See `psdist.bunch.histogram`.
    **kws
        Key word arguments passed to `plotting.image`.
    """
    if bins is None:
        bins = 50

    values, edges = np.histogramdd(points, bins=bins, range=limits)
    hist = Histogram(values, edges=edges)
    return _plot_hist(hist, ax=ax, **kws)


def plot_density_estimate(
    points: np.ndarray,
    bins: int = 10,
    limits: np.ndarray = None,
    method: str = "kde",
    method_kws: dict = None,
    ax=None,
    **kws,
):
    """Plot two-dimensional density estimate.

    Parameters
    ----------
    points: np.ndarray, shape (..., N)
        Particle coordinates.
    bins : int
       Number of bins defining the evaluation grid along each dimension.
    limits : np.ndarray
        Evaluation grid limits [(xmin, xmax), (ymin, ymax)].
    method: str
        Name of density estimator.
    method_kws: dict
        Key word arguments passed to density estimator.
    **kws
        Key word arguments passed to `plot_hist`.
    """
    if method_kws is None:
        method_kws = {}

    if limits is None:
        limits = get_limits(points)

    ndim = points.shape[1]

    edges = [np.linspace(limits[i][0], limits[i][1], bins + 1) for i in range(ndim)]
    hist = Histogram(edges=edges)
    eval_points = hist.points()

    values = _estimate_density(points, eval_points, method, **method_kws)
    values = values.reshape(hist.shape)
    hist.values = values
    hist.normalize()
    return _plot_hist(hist, ax=ax, **kws)


def plot(
    points: np.ndarray,
    kind: str = "hist",
    rms_ellipse: bool = False,
    rms_ellipse_kws: dict = None,
    ax=None,
    **kws,
):
    """Plot two-dimensional density.

    Parameters
    ----------
    points: np.ndarray, shape (..., N)
        Coordinates of points in N-dimensional space.
    kind : {'hist', 'contour', 'contourf', 'scatter', 'kde'}
        The kind of plot. These key words map to the following functions:
        "hist": psdist.plot.points.plot_hist(kind="hist")
        "contour": psdist.plot.points.plot_hist(kind="contour")
        "contourf": psdist.plot.points.plot_hist(kind="contourf")
        "scatter": psdist.plot.points.plot_scatter
        "density": psdist.plot.points.plot_density_estimate
    ax : Axes
        The axis on which to plot.
    **kws
        Key word arguments passed to plotting function.
    """
    if kind == "hist":
        kws.setdefault("mask", True)

    if rms_ellipse_kws is None:
        rms_ellipse_kws = {}

    plot_function = None
    if kind in ["hist", "contour", "contourf"]:
        plot_function = plot_hist
        if kind == "hist":
            kws["kind"] = "pcolor"
        if kind == "contour":
            kws["kind"] = "contour"
        if kind == "contourf":
            kws["kind"] = "contourf"
    elif kind == "scatter":
        plot_function = plot_scatter
    elif kind == "density":
        plot_function = plot_density_estimate
    else:
        raise ValueError(f"Invalid plot kind '{kind}'.")

    output = plot_function(points, ax=ax, **kws)
    if rms_ellipse:
        plot_rms_ellipse(points, ax=ax, **rms_ellipse_kws)
    return output


def plot_joint(points: np.ndarray, grid_kws: dict = None, **kws):
    from psdist.plot.grid import JointGrid

    if grid_kws is None:
        grid_kws = {}

    grid = JointGrid(**grid_kws)
    grid.plot_points(points, **kws)
    return grid


def plot_corner(points: np.ndarray, grid_kws: dict = None, **kws):
    from psdist.plot.grid import CornerGrid

    if grid_kws is None:
        grid_kws = {}

    ndim = points.shape[1]

    grid = CornerGrid(ndim=ndim, **grid_kws)
    grid.plot_points(points, **kws)
    return grid


def plot_interactive_slice_2d(
    data: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    limits: list[tuple[float, float]] = None,
    share_limits: int = 1,
    default_axis: tuple[int] = (0, 1),
    slice_type: str = "int",
    plot_res: int = 64,
    slice_res: int = 16,
    dims: list[str] = None,
    units: list[str] = None,
    options: dict = None,
    autolim_kws: dict = None,
    fig_kws: dict = None,
    **plot_kws,
):
    """Plot two-dimensional density with interactive slicing.

    Parameters
    ----------
    data : ndarray, shape (..., N) or list[ndarray] or list[list[ndarray]]
        ndarray:
            Corresponds to a single collection (bunch) of particles.
        list[ndarray]:
            Corresponds to a series of bunches. A widget is added to select the
            bunch to plot.
        list[list[ndarray]]:
            If there are K different lists provided, a K-column figure is produced.
            For example, if we had two different simulations, each generating a
            series of bunches at the same time steps, the function would create
            two subplots and a widget to select the frame.
    limits : list[tuple[float, float]]
        Limits (xmin, xmax) along each axis.
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
        Key word arguments passed to `plot`.
    """
    if type(data) is not list:
        data = [data]

    if type(data[0]) is not list:
        data = [data]

    nrows = len(data)
    ncols = len(data[0])
    ndims = data[0][0].shape[1]

    for i in range(nrows):
        if len(data[i]) != ncols:
            raise ValueError("lists must have same length")

    for i in range(nrows):
        for j in range(ncols):
            if data[i][j].shape[1] != ndims:
                raise ValueError("data must have the same number of dimensions.")

    if fig_kws is None:
        fig_kws = {}

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

    # Compute limits [(xmin, xmax), ...] for each bunch (data[i][j]).
    if autolim_kws is None:
        autolim_kws = {}

    if limits is None:
        limits_list = np.zeros((nrows, ncols, ndims, 2))
        for i in range(nrows):
            for j in range(ncols):
                limits_list[i, j, :, :] = get_limits(data[i][j], **autolim_kws)
        if share_limits == 1:
            for j in range(ncols):
                limits = combine_limits(limits_list[:, j, :, :])
                for i in range(nrows):
                    limits_list[i, j] = limits
        elif share_limits == 2:
            limits = combine_limits(limits_list.reshape((nrows * ncols, ndims, 2)))
            for i in range(nrows):
                for j in range(ncols):
                    limits_list[i, j] = limits
    else:
        limits_list = [[limits for _ in range(ncols)] for _ in range(nrows)]

    # Set axis labels.
    if dims is None:
        dims = [f"x{i + 1}" for i in range(ndims)]
    if units is None:
        units = ndims * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)

    # Widgets
    _widgets = {}
    _widgets["dim1"] = widgets.Dropdown(
        options=dims, index=default_axis[0], description="dim 1"
    )
    _widgets["dim2"] = widgets.Dropdown(
        options=dims, index=default_axis[1], description="dim 2"
    )
    _widgets["frame"] = widgets.BoundedIntText(
        min=0, max=(ncols - 1), description="frame"
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
    _widgets["mask"] = widgets.Checkbox(description="mask", value=False)
    _widgets["normalize"] = widgets.Checkbox(description="normalize", value=False)
    _widgets["profiles"] = widgets.Checkbox(description="profiles", value=False)

    # Sliders and checkboxes for slicing:
    _widgets["sliders"] = []
    _widgets["checks"] = []
    for k in range(ndims):
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
        for k in range(ndims):
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

    # Make slider visibility depend on checkmarks.
    for element in (
        _widgets["dim1"],
        _widgets["dim2"],
        *_widgets["checks"],
        _widgets["auto_plot_res"],
    ):
        element.observe(hide, names="value")

    # Initial hide
    for k in range(ndims):
        if k in default_axis:
            _widgets["checks"][k].layout.display = "none"
            _widgets["sliders"][k].layout.display = "none"
    if ncols == 1:
        _widgets["frame"].layout.display = "none"

    # Set default options.
    if options is None:
        options = {}
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

    plot_kws.setdefault("offset", 0.0)
    default_offset = plot_kws["offset"]

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
        for i in range(1, ndims + 1):
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

        # Collect data
        _data = [data[index][frame] for index in range(nrows)]
        _limits_list = limits_list[:, frame, :, :]

        # Normalize coordinates
        if kws["normalize"]:
            for i, _points in enumerate(_data):
                if _points.shape[1] % 2 == 0:
                    _data[i] = core.normalize_2d_projections(_points, scale=True)

            _limits_list = [core.limits(_points, **autolim_kws) for _points in _data]
            if share_limits > 0:
                _limits_list = [combine_limits(_limits_list) for _ in range(len(_data))]

        # Slice
        axis_view = [dims.index(dim) for dim in (dim1, dim2)]
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        if axis_slice:
            for index in range(nrows):
                slice_limits = []
                for k in axis_slice:
                    (imin, imax) = ind[k]
                    (xmin, xmax) = _limits_list[index][k]
                    edges = np.linspace(xmin, xmax, slice_res + 1)

                    if imax > len(edges) - 1:
                        print(f"{dims[k]} out of range.")
                        return

                    slice_limits.append((edges[imin], edges[imax]))

                _data[index] = core.slice_(
                    _data[index], axis=axis_slice, limits=slice_limits
                )

        # Handle empty slice (do nothing).
        for _points in _data:
            if _points.shape[0] == 0:
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
                if default_offset > 0.0:
                    plot_kws["offset"] = default_offset
                else:
                    plot_kws["offset"] = 1.0
            else:
                plot_kws["offset"] = default_offset

            # Temporary bug fix: If we check and then uncheck "log", and
            # the colorbar has minor ticks, the tick label formatter will
            # remain in "log" mode forever after.
            if "colorbar_kw" in plot_kws:
                if "tickminor" in plot_kws["colorbar_kw"] and not kws["log"]:
                    plot_kws["colorbar_kw"]["formatter"] = None

        ## Create figure.
        fig, axs = uplt.subplots(
            ncols=nrows,
            sharex=(share_limits and nrows > 1),
            sharey=(share_limits and nrows > 1),
            **fig_kws,
        )
        for index, ax in enumerate(axs):
            limits = [_limits_list[index][k] for k in axis_view]
            if plot_kws["kind"] != "scatter":
                plot_kws["limits"] = limits

            plot(_data[index][:, axis_view], ax=ax, **plot_kws)

            ax.format(xlim=limits[0], ylim=limits[1])

        labels = dims if _widgets["normalize"].value else dims_units
        axs.format(
            xlabel=labels[axis_view[0]],
            ylabel=labels[axis_view[1]],
        )
        plt.show()

    # Pass key word arguments to `ipywidgets.interactive`.
    kws = {}
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


def plot_interactive_slice_1d(
    data: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    limits: list[tuple[float]] = None,
    share_limits: int = 1,
    default_axis: int = 0,
    slice_type: str = "int",
    plot_res: int = 64,
    slice_res: int = 16,
    dims: list[str] = None,
    units: list[str] = None,
    options: dict = None,
    colors: list[str] = None,
    cycle: str = None,
    labels: list[str] = None,
    legend: bool = False,
    update_limits_on_slice: bool = False,
    legend_kws: dict = None,
    autolim_kws: dict = None,
    fig_kws: dict = None,
    **plot_kws,
):
    """One-dimensional partial projection of one or more bunches (or series of bunches)
    with interactive slicing.

    Profiles are scaled to unit area by default.

    Parameters
    ----------
    data : ndarray, shape (n, d) or list[ndarray] or list[list[ndarray]]
        - Particle coordinates.
        - List of L bunches: generates widget to select the frame to plot.
        - K lists of L bunches: generates K-column figure with widget to select one
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
        data = [data]

    nrows = len(data)
    ncols = len(data[0])
    ndims = data[0][0].shape[1]

    for i in range(nrows):
        if len(data[i]) != ncols:
            raise ValueError("lists must have same length")

    for i in range(nrows):
        for j in range(ncols):
            if data[i][j].shape[1] != ndims:
                raise ValueError("data must have the same number of dimensions.")

    if fig_kws is None:
        fig_kws = {}

    # Plot color cycle.
    if colors is None:
        if cycle is None:
            colors = uplt.Cycle(uplt.rc["cycle"]).by_key()["color"]
        else:
            colors = uplt.Cycle(cycle).by_key()["color"]
    if nrows == 1:
        for key in ["color", "c"]:
            if key in plot_kws:
                colors = ncols * [plot_kws.pop(key)]

    # Legend
    if legend_kws is None:
        legend_kws = {}
    legend_kws.setdefault("loc", "right")
    legend_kws.setdefault("framealpha", 0.0)
    legend_kws.setdefault("ncols", 1)

    if labels is None:
        labels = [f"prof_{i}" for i in range(nrows)]

    # Compute limits [(xmin, xmax), ...] for each frame.
    if autolim_kws is None:
        autolim_kws = {}
    if limits is None:
        limits_list = np.zeros((ncols, ndims, 2))
        for j in range(ncols):
            limits_list[j] = combine_limits(
                [core.limits(data[i][j], **autolim_kws) for i in range(nrows)]
            )
        if share_limits:
            limits = combine_limits(limits_list)
            for j in range(ncols):
                limits_list[j] = limits
    else:
        limits_list = [[limits for _ in range(ncols)] for _ in range(nrows)]

    # Set axis labels.
    if dims is None:
        dims = [f"x{i + 1}" for i in range(ndims)]
    if units is None:
        units = ndims * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)

    # Widgets
    _widgets = {}
    _widgets["dim"] = widgets.Dropdown(
        options=dims, index=default_axis, description="dim"
    )
    _widgets["frame"] = widgets.BoundedIntText(
        min=0, max=(ncols - 1), description="frame"
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
    _widgets["kde"] = widgets.Checkbox(desciption="kde", value=False)
    _widgets["log"] = widgets.Checkbox(description="log", value=False)
    _widgets["normalize"] = widgets.Checkbox(description="normalize", value=False)
    _widgets["scale"] = widgets.Dropdown(
        options=["none", "density", "max"], description="scale", value="density"
    )

    # Sliders and checkboxes for slicing:
    _widgets["sliders"] = []
    _widgets["checks"] = []
    for k in range(ndims):
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
        for k in range(ndims):
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

    # Make slider visibility depend on checkmarks.
    for element in [_widgets["dim"], *_widgets["checks"], _widgets["auto_plot_res"]]:
        element.observe(hide, names="value")

    # Initial hide
    for k in range(ndims):
        if k == default_axis:
            _widgets["checks"][k].layout.display = "none"
            _widgets["sliders"][k].layout.display = "none"
    if ncols == 1:
        _widgets["frame"].layout.display = "none"

    # Set default options.
    if options is None:
        options = {}
    options.setdefault("alpha", False)
    options.setdefault("auto_plot_res", False)
    options.setdefault("kde", False)
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
        for i in range(1, ndims + 1):
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
        _data = [data[index][frame] for index in range(nrows)]
        limits = limits_list[frame, :, :]

        # Normalize coordinates.
        if kws["normalize"]:
            for i, _points in enumerate(_data):
                if _points.shape[1] % 2 == 0:
                    _data[i] = core.normalize_2d_projections(_points, scale=True)

            limits = [core.limits(_points, **autolim_kws) for _points in _data]
            limits = combine_limits(limits)

        # Slice
        axis_view = dims.index(dim_view)
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        if axis_slice:
            for index in range(nrows):
                slice_limits = []
                for k in axis_slice:
                    (imin, imax) = ind[k]
                    (xmin, xmax) = limits[k]
                    edges = np.linspace(xmin, xmax, kws["slice_res"] + 1)
                    if imax > len(edges) - 1:
                        print(f"{dims[k]} out of range.")
                        return
                    slice_limits.append((edges[imin], edges[imax]))

                _data[index] = core.slice_(
                    _data[index], axis=axis_slice, limits=slice_limits
                )

        # Handle empty slice (do nothing).
        for _points in _data:
            if _points.shape[0] == 0:
                return

        # Recompute limits from sliced data.
        if update_limits_on_slice:
            limits = combine_limits(
                [core.limits(_points, **autolim_kws) for _points in _data]
            )

        # Plot
        fig, ax = uplt.subplots(**fig_kws)
        for index in range(len(_data)):
            plot_kws["color"] = colors[index]
            plot_kws["label"] = labels[index]
            plot_kws["alpha"] = kws["alpha"]
            plot_kws["kind"] = kws["kind"]
            plot_kws["scale"] = kws["scale"]
            plot_kws["kde"] = kws["kde"]

            bins = "auto" if kws["auto_plot_res"] else kws["plot_res"]

            x = _data[index][:, axis_view]
            plot_1d(x, bins=bins, limits=limits[axis_view], ax=ax, **plot_kws)

        if kws["log"]:
            ax.format(yscale="log", yformatter="log")

        ax.format(
            xlim=limits[axis_view],
            xlabel=(
                dims[axis_view]
                if _widgets["normalize"].value
                else dims_units[axis_view]
            ),
        )
        if legend and (labels is not None):
            ax.legend(**legend_kws)
        plt.show()

    # Pass key word arguments to `ipywidgets.interactive`.
    kws = {}
    for key in [
        "frame",
        "dim",
        "slice_res",
        "plot_res",
        "alpha",
        "auto_plot_res",
        "kind",
        "kde",
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
