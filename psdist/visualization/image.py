"""Plotting routines for multi-dimensional images."""
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt

import psdist.image
import psdist.utils
import psdist.visualization as vis


# TO DO
# - Update `proj2d_interactive_slice` and `proj1d_interactive_slice` to include
#   `options` input; see the cloud version of these functions.

def plot_profiles(
    f,
    coords=None,
    edges=None,
    ax=None,
    profx=True,
    profy=True,
    scale=0.12,
    start="edge",
    keep_limits=False,
    **kws,
):
    """Overlay one-dimensional profiles on a two-dimensional image.

    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    coords: (xcoords, ycoords)
        Lists specifying pixel center coordinates along each axis.
    coords: (xcoords, ycoords)
        Lists specifying pixel edge coordinates along each axis. Can be None.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    profx, profy : bool
        Whether to plot the x/y profile.
    scale : float
        Maximum of the 1D plot relative to the axes limits.
    start : {'edge', 'center'}
        Whether to start the plot at the center or edge of the first row/column.
    **kws
        Key word arguments passed to `psdist.visualization.plot_profile`.
    """
    kws.setdefault("kind", "step")
    kws.setdefault("lw", 1.0)
    kws.setdefault("color", "white")
    kws.setdefault("alpha", 0.6)
    if keep_limits:
        _limits = [ax.get_xlim(), ax.get_ylim()]
    if coords is None:
        if edges is not None:
            coords = [psdist.utils.centers_from_edges(e) for e in edges]
        else:
            coords = [np.arange(f.shape[k]) for k in range(f.ndim)]
    if edges is None:
        edges = [psdist.utils.edges_from_centers(c) for c in coords]
    for axis, proceed in enumerate([profx, profy]):
        if proceed:
            profile = psdist.image.project(f, axis=axis)
            profile_max = np.max(profile)
            if profile_max > 0.0:
                profile = profile / profile_max
            index = int(axis == 0)
            profile = profile * scale * np.abs(coords[index][-1] - coords[index][0])
            offset = 0.0
            if start == "edge":
                offset = edges[index][0]
            elif start == "center":
                offset = coords[index][0]
            psdist.visualization.plot_profile(
                profile=profile,
                coords=coords[axis],
                edges=edges[axis],
                ax=ax,
                offset=offset,
                orientation=("horizontal" if axis else "vertical"),
                **kws,
            )
        if keep_limits:
            ax.format(xlim=_limits[0], ylim=_limits[1])
    return ax


def plot_rms_ellipse(
    f, coords=None, ax=None, level=1.0, center_at_mean=True, **ellipse_kws
):
    """Compute and plot the RMS ellipse from a two-dimensional image.

    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    coords: (xcoords, ycoords)
        Lists specifying pixel center coordinates along each axis.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    level : number of list of numbers
        If a number, plot the rms ellipse inflated by the number. If a list
        of numbers, repeat for each number.
    center_at_mean : bool
        Whether to center the ellipse at the image centroid.
    """
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(f.ndim)]
    center = psdist.image.mean(f, coords) if center_at_mean else (0.0, 0.0)
    Sigma = psdist.image.cov(f, coords)
    return psdist.visualization.rms_ellipse(
        Sigma, center, level=level, ax=ax, **ellipse_kws
    )


def plot2d(
    f,
    coords=None,
    ax=None,
    kind="pcolor",
    profx=False,
    profy=False,
    prof_kws=None,
    process_kws=None,
    offset=None,
    offset_type="relative",
    mask=False,
    rms_ellipse=False,
    rms_ellipse_kws=None,
    return_mesh=False,
    **kws,
):
    """Plot a two-dimensional image.

    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    coords: (xcoords, ycoords)
        Lists specifying pixel center coordinates along each axis.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    kind : ['pcolor', 'contour', 'contourf']
        Whether to call `ax.pcolormesh`, `ax.contour`, or `ax.contourf`.
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
    process_kws : dict
        Key word arguments passed to `psdist.image.process`.
    offset, offset_type : float, {"relative", "absolute"}
        Adds offset to the image (helpful to get rid of zeros for logarithmic
        color scales. If offset_type is 'relative' add `min(f) * offset` to
        the image. Otherwise add `offset`.
    mask : bool
        Whether to plot pixels at or below zero.
    **kws
        Key word arguments passed to plotting function.
    """
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(f.ndim)]

    # Process key word arguments.
    if process_kws is None:
        process_kws = dict()

    if rms_ellipse_kws is None:
        rms_ellipse_kws = dict()

    func = None
    if kind == "pcolor":
        func = ax.pcolormesh
        kws.setdefault("ec", "None")
        kws.setdefault("linewidth", 0.0)
        kws.setdefault("rasterized", True)
        kws.setdefault("shading", "auto")
    elif kind == "contour":
        func = ax.contour
    elif kind == "contourf":
        func = ax.contourf
    else:
        raise ValueError("Invalid plot kind.")
    kws.setdefault("colorbar", False)
    kws.setdefault("colorbar_kw", dict())

    # Process the image.
    f = f.copy()
    f = psdist.image.process(f, **process_kws)
    if offset is not None:
        if offset_type == "relative" and np.count_nonzero(f):
            offset = offset * np.min(f[f > 0])
        f = f + offset
    else:
        offset = 0.0

    # Make sure there are no more zero elements if norm='log'.
    log = "norm" in kws and kws["norm"] == "log"
    if log:
        kws["colorbar_kw"]["formatter"] = "log"
    if mask or log:
        f = np.ma.masked_less_equal(f, 0)

    # If there are only zero elements, increase vmax so that the
    # lowest color is plotted.
    if not np.count_nonzero(f):
        kws["vmin"] = 1.0
        kws["vmax"] = 1.0

    # Plot the image.
    mesh = func(coords[0].T, coords[1].T, f.T, **kws)
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_rms_ellipse(f, coords=coords, ax=ax, **rms_ellipse_kws)
    if profx or profy:
        if prof_kws is None:
            prof_kws = dict()
        if kind == "contourf":
            prof_kws.setdefault("keep_limits", True)
        plot_profiles(
            f - offset, coords=coords, ax=ax, profx=profx, profy=profy, **prof_kws
        )
    if return_mesh:
        return ax, mesh
    else:
        return ax


def joint(f, coords=None, grid_kws=None, marg_kws=None, **kws):
    """Joint plot.

    This is a convenience function; see `psdist.visualization.grid.JointGrid`.

    Parameters
    ----------
    f : ndarray
        A 2-dimensional image.
    coords : list[ndarray]
        Coordinates along each dimension of `f`.
    grid_kws : dict
        Key word arguments passed to `JointGrid`.
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
    grid.plot_image(f, coords, marg_kws=marg_kws, **kws)
    return grid


def corner(
    f,
    coords=None,
    labels=None,
    prof_edge_only=False,
    update_limits=True,
    diag_kws=None,
    grid_kws=None,
    **kws,
):
    """Corner plot (scatter plot matrix).

    This is a convenience function; see `psdist.visualization.grid.CornerGrid`.

    Parameters
    ----------
    f : ndarray
        A d-dimensional image.
    coords : list[ndarray]
        Coordinates along each dimension of `f`.
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

    Returns
    -------
    psdist.visualization.grid.CornerGrid
        The `CornerGrid` on which the plot was drawn.
    """
    from psdist.visualization.grid import CornerGrid

    if grid_kws is None:
        grid_kws = dict()
    grid = CornerGrid(d=f.ndim, **grid_kws)
    if labels is not None:
        grid.set_labels(labels)
    grid.plot_image(
        f,
        coords=coords,
        prof_edge_only=prof_edge_only,
        update_limits=True,
        diag_kws=diag_kws,
        **kws,
    )
    return grid


def slice_matrix(
    f,
    coords=None,
    labels=None,
    axis_view=(0, 1),
    axis_slice=(2, 3),
    pad=0.0,
    debug=False,
    grid_kws=None,
    **kws,
):
    """Slice matrix plot.

    This is a convenience function; see `psdist.visualization.grid.SliceGrid`.

    Parameters
    ----------
    f : ndarray
        A d-dimensional image.
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
    grid_kws : dict
        Key word arguments passed to `visualization.grid.SliceGrid`.
    **kws
        Key word arguments passed to `visualization.image.plot2d`

    Returns
    -------
    psdist.visualization.grid.SliceGrid
        The `SliceGrid` on which the plot was drawn.
    """
    from psdist.visualization.grid import SliceGrid

    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault("space", 0.2)
    grid_kws.setdefault("annotate_kws_view", dict(color="white"))
    grid_kws.setdefault("annotate_kws_slice", dict(color="black"))
    grid_kws.setdefault("xticks", [])
    grid_kws.setdefault("yticks", [])
    grid_kws.setdefault("xspineloc", "neither")
    grid_kws.setdefault("yspineloc", "neither")
    grid = SliceGrid(**grid_kws)
    grid.plot_image(
        f,
        coords=None,
        labels=None,
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
    )
    return grid


def proj2d_interactive_slice(
    f,
    coords=None,
    default_ind=(0, 1),
    slice_type="int",
    dims=None,
    units=None,
    cmaps=None,
    thresh_slider=False,
    profiles_checkbox=False,
    **plot_kws,
):
    """2D partial projection with interactive slicing.

    The distribution is projected onto the specified axes. Sliders provide the
    option to slice the distribution before projecting.

    Parameters
    ----------
    f : ndarray
        A d-dimensional image.
    coords : list[ndarray]
        Coordinate arrays along each dimension. A square grid is assumed.
    default_ind : (i, j)
        Default x and y index to plot.
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims, units : list[str], shape (n,)
        Dimension names and units.
    cmaps : list
        Color map options for dropdown menu.
    thresh_slider : bool
        Whether to include a threshold slider.
    profiles_checkbox : bool
        Whether to include a profiles checkbox.
    **plot_kws
        Key word arguments for `visualization.image.plot2d`.

    Returns
    -------
    ipywidgets.widgets.interaction.interactive
        This widget can be displayed by calling `IPython.display.display(gui)`.
    """
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(f.ndim)]
    if dims is None:
        dims = [f"x{i + 1}" for i in range(f.ndim)]
    if units is None:
        units = f.ndim * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)
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
    if profiles_checkbox:
        _profiles_checkbox = widgets.Checkbox(value=False, description="profiles")
    else:
        _profiles_checkbox = None
    dim1 = widgets.Dropdown(options=dims, index=default_ind[0], description="dim 1")
    dim2 = widgets.Dropdown(options=dims, index=default_ind[1], description="dim 2")

    # Sliders and checkboxes (for slicing). Each unplotted dimension has a
    # checkbox which determine if that dimension is sliced. The slice
    # indices are determined by the slider.
    sliders, checks = [], []
    for k in range(f.ndim):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=f.shape[k],
                value=(f.shape[k] // 2),
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                value=(0, f.shape[k]),
                min=0,
                max=f.shape[k],
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
        for k in range(f.ndim):
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
    for k in range(f.ndim):
        if k in default_ind:
            checks[k].layout.display = "none"
            sliders[k].layout.display = "none"

    def update(**kws):
        """Update the figure."""
        dim1, dim2 = kws["dim1"], kws["dim2"]
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
        for k in range(f.ndim):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]
        idx = psdist.image.slice_idx(f.ndim, axis_slice, ind)
        _f = psdist.image.project(f[idx], axis_view)

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
        fig, ax = pplt.subplots()
        ax = plot2d(
            _f, coords=[coords[axis_view[0]], coords[axis_view[1]]], ax=ax, **plot_kws
        )
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()

    # Pass key word arguments for `update`.
    kws = dict()
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


def proj1d_interactive_slice(
    f,
    coords=None,
    default_ind=0,
    slice_type="int",
    dims=None,
    units=None,
    fig_kws=None,
    **plot_kws,
):
    """1D partial projection with interactive slicing.

    Parameters
    ----------
    f : ndarray
        A d-dimensional image.
    coords : list[ndarray]
        Grid coordinates for each dimension.
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
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(f.ndim)]
    if dims is None:
        dims = [f"x{i + 1}" for i in range(f.ndim)]
    if units is None:
        units = f.ndim * [""]
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f"{dim}" + f" [{unit}]" if unit != "" else dim)
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("figsize", (4.5, 1.5))
    plot_kws.setdefault("color", "black")
    plot_kws.setdefault("kind", "stepfilled")

    # Widgets
    dim1 = widgets.Dropdown(options=dims, index=default_ind, description="dim")

    # Sliders
    sliders, checks = [], []
    for k in range(f.ndim):
        if slice_type == "int":
            slider = widgets.IntSlider(
                min=0,
                max=f.shape[k],
                value=f.shape[k] // 2,
                description=dims[k],
                continuous_update=True,
            )
        elif slice_type == "range":
            slider = widgets.IntRangeSlider(
                value=(0, f.shape[k]),
                min=0,
                max=f.shape[k],
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
        for k in range(f.ndim):
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
    for k in range(f.ndim):
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
        for k in range(f.ndim):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]
        idx = psdist.image.slice_idx(f.ndim, axis_slice, ind)
        profile = psdist.image.project(f[idx], axis_view)

        # Make it a probability density function.
        profile = psdist.visualization.scale_profile(
            profile, coords=coords[axis_view], scale="density"
        )

        # Plot the projection.
        fig, ax = pplt.subplots(**fig_kws)
        ax.format(xlabel=dims_units[axis_view])
        psdist.visualization.plot_profile(
            profile=profile, coords=coords[axis_view], ax=ax, **plot_kws
        )
        plt.show()

    kws = {"dim1": dim1}
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)
