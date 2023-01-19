"""Plotting routines for 2n-dimensional phase space images."""
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt

import psdist.image
import psdist.visualization as vis
from psdist.utils import edges_from_centers


def plot_profiles(
    f,
    coords=None,
    ax=None,
    profx=True,
    profy=True,
    scale=0.12,
    start="edge",
    **kws,
):
    """Plot the one-dimensional profiles of a two-dimensional image.

    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    coords: (xcoords, ycoords)
        Lists specifying pixel center coordinates along each axis.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    profx, profy : bool
        Whether to plot the x/y profile.
    scale : float
        Maximum of the 1D plot relative to the axes limits.
    start : {'edge', 'center'}
        Whether to start the plot at the center or edge of the first row/column.
    **kws
        Key word arguments passed to `psdist.visualization.plot1d`.
    """
    kws.setdefault("kind", "step")
    kws.setdefault("lw", 1.0)
    kws.setdefault("color", "white")
    kws.setdefault("alpha", 0.6)

    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(f.ndim)]

    edges = [edges_from_centers(c) for c in coords]
    for axis, proceed in enumerate([profx, profy]):
        if proceed:
            profile = psdist.image.project(f, axis=axis)
            # Scale
            profile_max = np.max(profile)
            if profile_max > 0.0:
                profile = profile / profile_max
            j = int(axis == 0)
            profile = profile * scale * np.abs(coords[j][-1] - coords[j][0])
            # Start from ax spine.
            offset = 0.0
            if start == "edge":
                offset = edges[j][0]
            elif start == "center":
                offset = coords[j][0]
            vis.plot1d(coords[axis], profile, ax=ax, offset=offset, flipxy=axis, **kws)
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
    return vis.rms_ellipse(Sigma, center, level=level, ax=ax, **ellipse_kws)


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
    offset : float
        Adds `min(f) * offset` to the image. Helpful to get rid of zeros for
        logarithmic color scales.
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
        if np.count_nonzero(f):
            offset = offset * np.min(f[f > 0])
        f = f + offset

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

    # Plot.
    mesh = func(coords[0].T, coords[1].T, f.T, **kws)
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_rms_ellipse(f, coords=coords, ax=ax, **rms_ellipse_kws)
    if profx or profy:
        if prof_kws is None:
            prof_kws = dict()
        if kind == "contourf":
            prof_kws.setdefault("start", "center")
        plot_profiles(f, coords=coords, ax=ax, profx=profx, profy=profy, **prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax


def joint(f, coords=None, grid_kws=None, marg_kws=None, **kws):
    """Joint plot.
    
    This is a convenience function; see `psdist.visualization.JointGrid`.

    Parameters
    ----------
    f : ndarray
        An 2-dimensional image.
    coords : list[ndarray]
        Coordinates along each dimension of `f`.
    grid_kws : dict
        Key word arguments passed to `JointGrid`.
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

    This is a convenience function; see `psdist.visualization.CornerGrid`.

    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
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
    psdist.visualization.CornerGrid
        The `CornerGrid` on which the plot was drawn.
    """
    if grid_kws is None:
        grid_kws = dict()
    cgrid = vis.CornerGrid(n=f.ndim, **grid_kws)
    if labels is not None:
        cgrid.set_labels(labels)
    cgrid.plot_image(
        f,
        coords=coords,
        prof_edge_only=prof_edge_only,
        update_limits=True,
        diag_kws=diag_kws,
        **kws,
    )
    return cgrid


def slice_matrix(
    f,
    coords=None,
    labels=None,
    axis_view=(0, 1),
    axis_slice=(2, 3),
    pad=0.0,
    debug=False,
    **kws,
):
    """Slice matrix plot.

    This is a convenience function; see `psdist.visualization.SliceGrid`.

    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
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

    Returns
    -------
    psdist.visualization.SliceGrid
        The `SliceGrid` on which the plot was drawn.
    """
    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault("space", 0.2)
    grid_kws.setdefault("annotate_kws_view", dict(color="white"))
    grid_kws.setdefault("annotate_kws_slice", dict(color="black"))
    grid_kws.setdefault("xticks", [])
    grid_kws.setdefault("yticks", [])
    grid_kws.setdefault("xspineloc", "neither")
    grid_kws.setdefault("yspineloc", "neither")
    sgrid = psv.SliceGrid(**grid_kws)
    sgrid.plot_image(
        f,
        coords=None,
        labels=None,
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
    )
    return sgrid


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
    """2D partial projection of image with interactive slicing.

    The distribution is projected onto the specified axes. Sliders provide the
    option to slice the distribution before projecting.

    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
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
    n = f.ndim
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(n)]
    if dims is None:
        dims = [f"x{i + 1}" for i in range(n)]
    if units is None:
        units = n * [""]
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
    for k in range(n):
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

    # Update the slider list automatically.
    for element in (dim1, dim2, *checks):
        element.observe(hide, names="value")
    # Initial hide
    for k in range(n):
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
        for k in range(n):
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
    """1D partial projection of n-dimensional image with interactive slicing.

    Parameters
    ----------
    f : ndarray
        An n-dimensional image.
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
    n = f.ndim
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(n)]
    if dims is None:
        dims = [f"x{i + 1}" for i in range(n)]
    if units is None:
        units = n * [""]
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
    for k in range(n):
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
        for k in range(n):
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
    for k in range(n):
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
        for k in range(n):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]
        idx = psdist.image.slice_idx(f.ndim, axis_slice, ind)
        profile = psdist.image.project(f[idx], axis_view)
        if np.max(profile) > 0:
            profile = profile / np.sum(profile)
        # Plot the projection.
        fig, ax = pplt.subplots(**fig_kws)
        ax.format(xlabel=dims_units[axis_view])
        vis.plot1d(coords[axis_view], profile, ax=ax, **plot_kws)
        plt.show()

    kws = {"dim1": dim1}
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)
