"""Plotting routines for multi-dimensional images."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import ultraplot as uplt
from ipywidgets import interactive
from ipywidgets import widgets

import psdist as ps
from ..hist import Histogram
from ..hist import Histogram1D
from . import core


def _process_values_fill(values: np.ndarray, fill_value: float) -> np.ndarray:
    return np.ma.filled(values, fill_value=fill_value)


def _process_values_thresh(
    values: np.ndarray, lmin: 0.0, frac: bool = False
) -> np.ndarray:
    if frac:
        lmin = lmin * values.max()
    values[values < lmin] = 0.0
    return values


def _process_values_clip(
    values: np.ndarray, lmin: float = None, lmax: float = None, frac: bool = False
) -> np.ndarray:
    """Clip between lmin and lmax, can be fractions or absolute values."""
    if not (lmin or lmax):
        return values
    if frac:
        f_max = np.max(f)
        if lmin:
            lmin = f_max * lmin
        if lmax:
            lmax = f_max * lmax
    return np.clip(f, lmin, lmax)


def _process_values_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    return scipy.ndimage.gaussian_filter(values, sigma)


def _process_values_scale_max(values: np.ndarray) -> np.ndarray:
    values_max = np.max(values)
    if values_max > 0.0:
        values = values / values_max
    return values


def process_hist(
    hist: Histogram | Histogram1D,
    fill_value: float = None,
    thresh: float = None,
    thresh_type: str = "abs",
    clip: tuple[float] = None,
    clip_type: str = "abs",
    blur: float = None,
    scale: str = None,
) -> Histogram:
    """Return processed image.

    Defaults leave histogram unchanged.

    Parameters
    ----------
    hist : Histogram
        A two-dimensional histogram.
    fill_value : float
        Fill masked elements of `f` with this value.
    thresh : float
        Set elements below this value to zero.
    thresh_type : {"frac", "abs"}
        Whether `thresh` is absolute or relative to peak.
    clip : (lmin, lmax)
        Clip (limit) elements to within the range [lmin, lmax].
    clip_type : {"frac", "abs"}
        Whether `clip` is absolute or relative to peak.
    scale : {None, "max", "density"}
        Normalize by max, volume, or None.
    blur : float
        Kernel width for gaussian filter.
    """
    values = hist.values.copy()

    if fill_value is not None:
        values = _process_values_fill(values, fill_value)
    if thresh is not None:
        values = _process_values_thresh(values, thresh, frac=(thresh_type == "frac"))
    if clip is not None:
        values = _process_values_clip(
            values, lmin=clip[0], lmax=clip[1], frac=(clip_type == "fract")
        )
    if blur is not None:
        values = _process_values_blur(values, blur)
    if scale == "max":
        values = _process_values_scale_max(values)
    hist.values = values
    if scale == "density":
        hist.normalize()
    return hist


def scale_hist(
    hist: Histogram1D | Histogram, scale: str | float | int = None
) -> Histogram1D | Histogram:
    if scale is None:
        return hist

    if np.max(hist.values) <= 0.0:
        return hist

    if np.sum(hist.values) <= 0.0:
        return hist

    normalization = 1.0
    if scale == "density":
        normalization = np.sum(hist.values) * hist.cell_volume
    elif scale == "max":
        normalization = np.max(hist.values)
    else:
        normalization = scale

    if normalization > 0.0:
        hist.values = hist.values / normalization
    return hist


def plot_1d(
    hist: Histogram1D,
    orientation: str = "vertical",
    kind: str = "line",
    fill: bool = False,
    offset: float = 0.0,
    scale: float = None,
    process_kws: dict = None,
    ax=None,
    **kws,
):
    """Plot one-dimensional histogram.

    Parameters
    ----------
    hist : Histogram1D
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
    process_kws : dict
        Key word arguments passed to `process_hist`.
    **kws
        Key word arguments passed to the plotting function.
    """
    if process_kws is None:
        process_kws = {}

    kws.setdefault("lw", 1.5)

    hist = hist.copy()
    hist = scale_hist(hist, scale=scale)

    values = hist.values.copy()
    coords = hist.coords.copy()
    edges = hist.edges.copy()

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
        values = values + offset
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
            return ax.bar(coords, values, bottom=pad, **kws)
    else:
        raise ValueError(f"Invalid plot kind '{kind}'")

    return ax


def plot_hist_profiles_overlay(
    hist: Histogram,
    profx: bool = True,
    profy: bool = True,
    scale: float = 0.12,
    start: str = "edge",
    keep_limits: bool = False,
    ax=None,
    **kws,
) -> uplt.Axes:
    """Overlay one-dimensional profiles on two-dimensional image.

    Parameters
    ----------
    hist : Histogram
        A two-dimensional histogram.
    profx, profy : bool
        Whether to plot the x/y profile.
    scale : float
        Maximum of the 1D plot relative to the axes limits.
    start : {'edge', 'center'}
        Whether to start the plot at the center or edge of the first row/column.
    **kws
        Key word arguments passed to `psdist.plot.plot_profile`.
    """
    kws.setdefault("kind", "step")
    kws.setdefault("lw", 1.0)
    kws.setdefault("color", "white")  # good for viridis
    kws.setdefault("alpha", 0.6)

    values = hist.values.copy()
    coords = hist.coords

    old_limits = [ax.get_xlim(), ax.get_ylim()]

    for axis, proceed in enumerate([profx, profy]):
        if proceed:
            hist_proj = hist.project(axis)
            hist_proj = scale_hist(hist_proj, scale="max")

            # Scale values based on coordinates on other axis
            other_axis = int(axis == 0)
            hist_proj.values *= scale * hist.ranges[other_axis]

            offset = 0.0
            if start == "edge":
                offset = hist.edges[other_axis][0]
            elif start == "center":
                offset = hist.coords[other_axis][0]

            plot_1d(
                hist=hist_proj,
                ax=ax,
                offset=offset,
                orientation=("horizontal" if axis else "vertical"),
                **kws,
            )
        if keep_limits:
            ax.format(xlim=old_limits[0], ylim=old_limits[1])
    return ax


def plot_rms_ellipse(
    hist: Histogram,
    level: float = 1.0,
    center_at_mean: bool = True,
    ax=None,
    **kws,
) -> uplt.Axes:
    """Compute and plot the RMS ellipse from a two-dimensional image.

    Parameters
    ----------
    hist : Histogram
        A two-dimensional histogram.
    level : number of list of numbers
        If a number, plot the rms ellipse inflated by the number. If a list
        of numbers, repeat for each number.
    center_at_mean : bool
        Whether to center the ellipse at the image centroid.
    **kws
        Key word arguments passed to `psdist.plot.plot_rms_ellipse`.
    """
    cov_matrix = ps.hist.cov(hist)
    mean = (0.0, 0.0)
    if center_at_mean:
        mean = ps.hist.mean(hist)
    return core.plot_rms_ellipse_cov(cov_matrix, mean, level=level, ax=ax, **kws)


def plot(
    hist: Histogram,
    kind: str = "pcolor",
    process_kws: dict = None,
    offset: float = None,
    offset_type: str = "relative",
    mask: bool = False,
    profx: bool = False,
    profy: bool = False,
    prof_kws: dict = None,
    rms_ellipse: bool = False,
    rms_ellipse_kws: dict = None,
    return_mesh: bool = False,
    ax=None,
    **kws,
) -> uplt.Axes:
    """Plot two-dimensional histogram..

    Parameters
    ----------
    hist : Histogram
        A two-dimensional histogram.
    kind : ['pcolor', 'contour', 'contourf']
        Whether to call `ax.pcolormesh`, `ax.contour`, or `ax.contourf`.
    process_kws : dict
        Key word arguments passed to `psdist.image.process`.
    offset: float
        Adds offset to the histogram values (usually to avoid zeros for
         logarithmic colormaps).
    offset_type : {"relative", "absolute"}
        If offset_type is 'relative', adds `min(hist.values) * offset`.
        Otherwise, adds `offset`.
    mask : bool
        Whether to mask zero/negative cells.
    profx, profy : bool
        Whether to plot the x/y profile.
    prof_kws : dict
        Key words arguments for `image_profiles`.
    rms_ellipse : bool
        Whether to plot rms ellipse.
    rms_ellipse_kws : dict
        Key word arguments for `image_rms_ellipse`.
    return_mesh : bool
        Whether to return a mesh from `ax.pcolormesh`.
    ax : Axes
        The axis on which to plot.
    **kws
        Key word arguments passed to plotting function.
    """
    # Process key word arguments.
    if process_kws is None:
        process_kws = {}

    if rms_ellipse_kws is None:
        rms_ellipse_kws = {}
    rms_ellipse_kws.setdefault("color", "white")

    if prof_kws is None:
        prof_kws = {}
    if kind == "contourf":
        prof_kws.setdefault("keep_limits", True)

    kws.setdefault("colorbar", False)
    kws.setdefault("colorbar_kw", {})

    plot_function = None
    if kind == "pcolor":
        plot_function = ax.pcolormesh
        kws.setdefault("ec", "None")
        kws.setdefault("linewidth", 0.0)
        kws.setdefault("rasterized", True)
        kws.setdefault("shading", "auto")
    elif kind == "contour":
        plot_function = ax.contour
    elif kind == "contourf":
        plot_function = ax.contourf
    else:
        raise ValueError("Invalid plot kind.")

    # Process the histogram.
    hist = hist.copy()
    hist = process_hist(hist, **process_kws)

    # Add offset to histogram values.
    if offset is not None:
        if offset_type == "relative":
            if np.count_nonzero(hist.values):
                offset *= np.min(hist.values[hist.values > 0])
        hist.values += offset
    else:
        offset = 0.0

    # Make sure there are no more zero elements if kws["norm"] == "log"
    log = ("norm" in kws) and (kws["norm"] == "log")
    if mask or log:
        hist.values = np.ma.masked_less_equal(hist.values, 0.0)
    if log:
        kws["colorbar_kw"]["formatter"] = "log"

    # If there are only zero elements, increase vmax so that the lowest color shows.
    if not np.count_nonzero(hist.values):
        kws["vmin"] = 1.0
        kws["vmax"] = 1.0

    # Plot the image.
    mesh = plot_function(hist.coords[0], hist.coords[1], hist.values.T, **kws)
    if rms_ellipse:
        plot_rms_ellipse(hist, ax=ax, **rms_ellipse_kws)
    if profx or profy:
        hist.values -= offset
        plot_hist_profiles_overlay(hist, profx=profx, profy=profy, ax=ax, **prof_kws)

    if return_mesh:
        return ax, mesh
    else:
        return ax


def plot_joint(hist: Histogram, grid_kws: dict = None, **kws):
    from psdist.plot.grid import JointGrid

    if grid_kws is None:
        grid_kws = {}

    grid = JointGrid(**grid_kws)
    grid.plot_hist(hist, **kws)
    return grid


def plot_corner(hist: Histogram, grid_kws: dict = None, **kws):
    from psdist.plot.grid import CornerGrid

    if grid_kws is None:
        grid_kws = {}

    ndim = hist.ndim

    grid = CornerGrid(ndim=ndim, **grid_kws)
    grid.plot_hist(hist, **kws)
    return grid


def plot_slice_matrix(hist: Histogram, grid_kws: dict = None, **kws):
    from psdist.plot.grid import SliceGrid

    if grid_kws is None:
        grid_kws = {}
    grid_kws.setdefault("space", 0.2)
    grid_kws.setdefault("annotate_kws_view", dict(color="white"))
    grid_kws.setdefault("annotate_kws_slice", dict(color="black"))
    grid_kws.setdefault("xticks", [])
    grid_kws.setdefault("yticks", [])
    grid_kws.setdefault("xspineloc", "neither")
    grid_kws.setdefault("yspineloc", "neither")

    grid = SliceGrid(**grid_kws)
    grid.plot_hist(hist, **kws)
    return grid


# # [TO DO] Update `proj2d_interactive_slice` and `proj1d_interactive_slice` to include
# # `options` input; see the versions of these functions in `psdist.plot.points`.
#


def plot_interactive_slice_2d(
    hist: Histogram,
    default_ind: tuple[int, int] = (0, 1),
    slice_type: str = "int",
    dims: list[str] = None,
    units: list[str] = None,
    cmaps: list[str] = None,
    thresh_slider: bool = False,
    profiles_checkbox: bool = False,
    fig_kws: dict = None,
    **plot_kws,
):
    """2D partial projection with interactive slicing.

    The distribution is projected onto the specified axes. Sliders provide the
    option to slice the distribution before projecting.

    Parameters
    ----------
    hist : Histogram
        A two-dimensional histogram.
    default_ind : (i, j)
        Default x and y index to plot.
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    fig_kws : dict
        Key words for `uplt.subplots`.
    cmaps : list
        Color map options for dropdown menu.
    thresh_slider : bool
        Whether to include a threshold slider.
    profiles_checkbox : bool
        Whether to include a profiles checkbox.
    **plot_kws
        Key word arguments for `plot.image.plot`.

    Returns
    -------
    ipywidgets.widgets.interaction.interactive
        This widget can be displayed by calling `IPython.display.display(gui)`.
    """
    if fig_kws is None:
        fig_kws = {}

    if dims is None:
        dims = [f"x{i + 1}" for i in range(hist.ndim)]

    if units is None:
        units = hist.ndim * [""]

    dims_units = []
    for dim, unit in zip(dims, units):
        if unit:
            dims_units.append(f"{dim} [{unit}]")
        else:
            dims_units.append(dim)

    plot_kws.setdefault("colorbar", True)
    plot_kws["process_kws"] = dict(thresh_type="frac")

    # Widgets
    cmap = None
    if cmaps is not None:
        cmap = widgets.Dropdown(options=cmaps, description="cmap")

    if thresh_slider:
        thresh_slider = widgets.FloatSlider(
            value=-3.3,
            min=-8.0,
            max=0.0,
            step=0.1,
            description="thresh (log)",
            continuous_update=True,
        )

    discrete = widgets.Checkbox(value=False, description="discrete")

    log = widgets.Checkbox(value=False, description="log")

    _profiles_checkbox = None
    if profiles_checkbox:
        _profiles_checkbox = widgets.Checkbox(value=False, description="profiles")

    dim1 = widgets.Dropdown(options=dims, index=default_ind[0], description="dim 1")
    dim2 = widgets.Dropdown(options=dims, index=default_ind[1], description="dim 2")

    # Sliders and checkboxes (for slicing). Each unplotted dimension has a
    # checkbox which determine if that dimension is sliced. The slice
    # indices are determined by the slider.
    sliders, checks = [], []
    for k in range(hist.ndim):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=hist.shape[k],
                value=(hist.shape[k] // 2),
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                value=(0, hist.shape[k]),
                min=0,
                max=hist.shape[k],
                description=dims[k],
                continuous_update=True,
            )
        else:
            raise ValueError("`slice_type` must be 'int' or 'range'.")

        slider.layout.display = "none"
        sliders.append(slider)
        checks.append(widgets.Checkbox(description=f"slice {dims[k]}"))

    def hide(button):
        """Hide/show sliders."""
        for k in range(hist.ndim):
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

    # Update the slider list automatically.
    for element in (dim1, dim2, *checks):
        element.observe(hide, names="value")

    # Initial hide
    for k in range(hist.ndim):
        if k in default_ind:
            checks[k].layout.display = "none"
            sliders[k].layout.display = "none"

    def update(**kws):
        """Update the figure."""
        dim1 = kws["dim1"]
        dim2 = kws["dim2"]

        ind, checks = [], []
        for i in range(100):
            if f"check{i}" in kws:
                checks.append(kws[f"check{i}"])
            if f"slider{i}" in kws:
                ind.append(kws[f"slider{i}"])

        # Return nothing if input does not make sense.
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        if dim1 == dim2:
            return

        # Slice and project the distribution.
        axis_view = [dims.index(dim) for dim in (dim1, dim2)]
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        for k in range(hist.ndim):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]

        hist_proj = hist.slice(axis=axis_slice, ind=ind).project(axis_view)

        # Update plotting key word arguments.
        if "cmap" in kws:
            plot_kws["cmap"] = kws["cmap"]
        if "profiles_checkbox" in kws:
            plot_kws["profx"] = plot_kws["profy"] = kws["profiles_checkbox"]
        plot_kws["norm"] = "log" if kws["log"] else None
        plot_kws["process_kws"]["fill_value"] = 0
        if "thresh_slider" in kws:
            plot_kws["process_kws"]["thresh"] = 10.0 ** kws["thresh_slider"]
        else:
            plot_kws["process_kws"]["thresh"] = None

        # Plot the projection onto the specified axes.
        fig, ax = uplt.subplots(**fig_kws)
        plot(hist_proj, ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()

    # Pass key word arguments for `update`.
    kws = {}
    if cmap is not None:
        kws["cmap"] = cmap
    if profiles_checkbox:
        kws["profiles_checkbox"] = _profiles_checkbox
    kws["log"] = log
    kws["dim1"] = dim1
    kws["dim2"] = dim2
    if thresh_slider:
        kws["thresh_slider"] = thresh_slider
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)


def plot_interactive_slice_1d(
    hist: Histogram,
    default_ind: int = 0,
    slice_type: str = "int",
    dims: list[str] = None,
    units: list[str] = None,
    fig_kws: dict = None,
    **plot_kws,
):
    """1D partial projection with interactive slicing.

    Parameters
    ----------
    hist : Histogram
        A two-dimensional histogram.
    default_ind : int
        Default index to plot.
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    kind : {'bar', 'line'}
        The kind of plot to draw.
    fig_kws : dict
        Key word arguments passed to `proplot.subplots`.
    **plot_kws
        Key word arguments passed to 1D plotting function.

    Returns
    -------
    gui : ipywidgets.widgets.interaction.interactive
        This widget can be displayed by calling `IPython.display.display(gui)`.
    """
    if dims is None:
        dims = [f"x{i + 1}" for i in range(hist.ndim)]

    if units is None:
        units = hist.ndim * [""]

    dims_units = []
    for dim, unit in zip(dims, units):
        if unit:
            dims_units.append(f"{dim} [{unit}]")
        else:
            dims_units.append(dim)

    if fig_kws is None:
        fig_kws = {}
    fig_kws.setdefault("figsize", (4.5, 1.5))

    plot_kws.setdefault("color", "black")
    plot_kws.setdefault("kind", "stepfilled")

    # Widgets
    dim1 = widgets.Dropdown(options=dims, index=default_ind, description="dim")

    # Sliders
    sliders, checks = [], []
    for k in range(hist.ndim):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=hist.shape[k],
                value=hist.shape[k] // 2,
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                value=(0, hist.shape[k]),
                min=0,
                max=hist.shape[k],
                description=dims[k],
                continuous_update=True,
            )
        else:
            raise ValueError("Invalid `slice_type`.")
        slider.layout.display = "none"
        sliders.append(slider)
        checks.append(widgets.Checkbox(description=f"slice {dims[k]}"))

    def hide(button):
        """Hide/show sliders based on checkboxes."""
        for k in range(hist.ndim):
            # Hide elements for dimensions being plotted.
            valid = dims[k] != dim1.value
            disp = None if valid else "none"
            for element in [sliders[k], checks[k]]:
                element.layout.display = disp

            # Uncheck boxes for dimensions being plotted.
            if not valid and checks[k].value:
                checks[k].value = False

            # Make sliders respond to check boxes.
            if not checks[k].value:
                sliders[k].layout.display = "none"

    # Update the slider list automatically.
    for element in (dim1, *checks):
        element.observe(hide, names="value")

    # Initial hide
    for k in range(hist.ndim):
        if k == default_ind:
            checks[k].layout.display = "none"
            sliders[k].layout.display = "none"

    def update(**kws):
        """Update the figure."""
        dim1 = kws["dim1"]

        ind, checks = [], []
        for i in range(100):
            if f"check{i}" in kws:
                checks.append(kws[f"check{i}"])
            if f"slider{i}" in kws:
                ind.append(kws[f"slider{i}"])

        # Return nothing if input does not make sense.
        for dim, check in zip(dims, checks):
            if check and dim == dim1:
                return

        # Slice, then project onto the specified axis.
        axis_view = dims.index(dim1)
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        for k in range(hist.ndim):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]

        hist_proj = hist.slice(axis=axis_slice, ind=ind).project(axis_view)
        hist_proj.normalize()

        # Plot the projection.
        fig, ax = uplt.subplots(**fig_kws)
        plot_1d(hist_proj, ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view])
        plt.show()

    kws = {"dim1": dim1}
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)
