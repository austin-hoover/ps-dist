"""Plotting routines for 2n-dimensional phase space images."""
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt

from .. import image as psi
from . import visualization as vis
from ..utils import edges_from_centers


def plot_profiles(
    f,
    coords=None,
    ax=None,
    profx=True,
    profy=True,
    scale=0.12,
    start='edge',
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
    kws.setdefault('kind', 'step')
    kws.setdefault('lw', 1.0)
    kws.setdefault('color', 'white')
    kws.setdefault('alpha', 0.6)
    
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(f.ndim)]

    edges = [edges_from_centers(c) for c in coords]
    for axis, proceed in enumerate([profx, profy]):
        if proceed:    
            profile = psi.project(f, axis=axis)
            # Scale
            profile_max = np.max(profile)
            if profile_max > 0.0:
                profile = profile / profile_max
            j = int(axis == 0)            
            profile = profile * scale * np.abs(coords[j][-1] - coords[j][0])
            # Start from ax spine.
            offset = 0.0
            if start == 'edge':
                offset = edges[j][0]
            elif start == 'center':
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
    center = psi.mean(f, coords) if center_at_mean else (0.0, 0.0)
    Sigma = psi.cov(f, coords)
    return vis.rms_ellipse(Sigma, center, level=level, ax=ax, **ellipse_kws)


def plot2d(
    f,
    coords=None,
    ax=None,
    kind='pcolor',
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
    if kind == 'pcolor':
        func = ax.pcolormesh
        kws.setdefault('ec', 'None')
        kws.setdefault('linewidth', 0.0)
        kws.setdefault('rasterized', True)
    elif kind == 'contour':
        func = ax.contour
    elif kind == 'contourf':
        func = ax.contourf
    else:
        raise ValueError('Invalid plot kind.')
    kws.setdefault('colorbar', False)
    kws.setdefault('colorbar_kw', dict())
    
    # Process the image.
    f = f.copy()
    f = psi.process(f, **process_kws)
    if offset is not None:
        if np.count_nonzero(f):
            offset = offset * np.min(f[f > 0])
        f = f + offset
        
    # Make sure there are no more zero elements if norm='log'.
    log = 'norm' in kws and kws['norm'] == 'log'
    if log:
        kws['colorbar_kw']['formatter'] = 'log'
    if mask or log:
        f = np.ma.masked_less_equal(f, 0)
           
    # Plot.
    mesh = func(coords[0].T, coords[1].T, f.T, **kws)        
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_rms_ellipse(f, coords=coords, ax=ax, **rms_ellipse_kws)
    if profx or profy:
        if prof_kws is None:
            prof_kws = dict()
        if kind == 'contourf':
            prof_kws.setdefault('start', 'center')
        plot_profiles(f, coords=coords, ax=ax, profx=profx, profy=profy, **prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax
    
    
def jointplot():
    # This will be like seaborn.jointplot (top/right panel axes).
    raise NotImplementedError
    
    
def corner():
    # Corner plot for image.
    raise NotImplementedError
    
    
def slice_matrix(
    f,
    axis_view=None,
    axis_slice=None,
    nrows=9,
    ncols=9,
    coords=None,
    dims=None,
    space=0.1,
    gap=2.0,
    pad=0.0,
    fig_kws=None,
    plot_marginals=True,
    plot_kws_marginal_only=None,
    return_indices=False,
    annotate=True,
    label_height=0.22,
    debug=False,
    **plot_kws,
):
    """Matrix of 2D projections as two other dimensions are sliced.

    In the following, assume `axis_slice`=(0, 1) and `axis_view=(2, 3)`:

    First, `f` is projected onto the (0, 1, 2, 3) axes. The remaining 4D
    array is sliced using `ncols` evenly spaced indices along axis 0 and
    `nrows` evenly spaced indices along axis 1. The resulting array has shape
    (`ncols`, `nrows`, `f.shape[2]`, `f.shape[3]`). For i in range(`ncols`) and
    j in range(`nrows`), we plot the 2D image `f[i, j, :, :]`. This is done in
    a matrix of subplots in the upper-left panel.

    Second, 2D slices of the 3D array obtained by summing `f` along axis 0 are
    plotted in the upper-right panel.

    Third, 2D slices of the 3D array obtained by summing `f` along axis 1 are
    plotted in the lower-left panel.

    Fourth, `f` is projected onto axis (2, 3) and plotted in the lower-right p
    panel.

    Parameters
    ----------
    f : ndarray
        An n-dimensional image (n >= 4).
    axis_view : 2-tuple of int
        The dimensions to plot.
    axis_slice : 2-tuple of int
        The dimensions to slice.
    nrows, ncols : int
        The number of slices along each axis in `axis_slice`.
    coords : list[ndarray]
        Coordinates along each axis of the grid (if `data` is an image).
    dims : list[str]
        Labels for each dimension.
    space : float
        Spacing between subplots.
    gap : float
        Gap between major panels.
    pad : int, float, list
        This determines the start/stop indices along the sliced dimensions. If
        0, space the indices along axis `i` uniformly between 0 and `f.shape[i]`.
        Otherwise, add a padding equal to `int(pad[i] * f.shape[i])`. So, if
        the shape=10 and pad=0.1, we would start from 1 and end at 9.
    fig_kws : dict
        Key word arguments for `pplt.subplots`.
    plot_marginals : bool
        Whether to plot the 3D and 2D marginal distributions.
    plot_kws_marginal_only : dict
        Key word arguments for the lower-left and upper-right panels, which
        plot the 3D marginal distributions.
    return_indices : bool
        Whether to return the slice indices.
    annotate : bool
        Whether to add dimension labels/arrows to the figure.
    label_height : float
        Tweaks the position of the slice dimension labels.
    debug : bool
        Whether to print debugging messages.
    **plot_kws
        Key word arguments for `image`.

    Returns
    -------
    axes : Axes
        The plot axes.
    ind_slice : 2-tuple of lists (optional)
        The slice indices along `axis_slice`. Returned if `return_indices` is True.
    """
    # Setup
    # -------------------------------------------------------------------------
    if f.ndim < 4:
        raise ValueError("f.ndim < 4")
    if axis_view is None:
        axis_view = (0, 1)
    if axis_slice is None:
        axis_slice = (2, 3)
    if coords is None:
        coords = [np.arange(s) for s in f.shape]

    # Compute 4D projection.
    _f = psi.project(f, axis_view + axis_slice)
    _f = _f / np.max(_f)
    # Compute 3D projections.
    _fx = psi.project(f, axis_view + axis_slice[:1])
    _fy = psi.project(f, axis_view + axis_slice[1:])
    # Compute 2D projection.
    _fxy = psi.project(f, axis_view)
    # Compute new coordinates.
    _coords = [coords[i] for i in axis_view + axis_slice]
    # Compute new dims.
    _dims = None
    if dims is not None:
        _dims = [dims[i] for i in axis_view + axis_slice]

    # Select slice indices.
    if type(pad) in [float, int]:
        pad = len(axis_slice) * [pad]
    ind_slice = []
    for i, n, _pad in zip(axis_slice, [nrows, ncols], pad):
        s = f.shape[i]
        _pad = int(_pad * s)
        ii = np.linspace(_pad, s - 1 - _pad, n).astype(int)
        ind_slice.append(ii)
    ind_slice = tuple(ind_slice)

    if debug:
        print("Slice indices:")
        for ind in ind_slice:
            print(ind)

    # Slice _f. The axes order was already handled by `project`, so the
    # first two axes are the view axes and the last two axes are the
    # slice axes.
    axis_view = (0, 1)
    axis_slice = (2, 3)
    idx = 4 * [slice(None)]
    for axis, ind in zip(axis_slice, ind_slice):
        idx[axis] = ind
        _f = _f[tuple(idx)]
        idx[axis] = slice(None)

    # Slice _fx and _fy.
    _fx = _fx[:, :, ind_slice[0]]
    _fy = _fy[:, :, ind_slice[1]]

    # Select new coordinates.
    for i, ind in zip(axis_slice, ind_slice):
        _coords[i] = _coords[i][ind]

    # Renormalize all distributions.
    _f = _f / np.max(_f)
    _fx = _fx / np.max(_fx)
    _fy = _fy / np.max(_fy)
    _fxy = _fxy / np.max(_fxy)

    if debug:
        print("_f.shape =", _f.shape)
        print("_fx.shape =", _fx.shape)
        print("_fy.shape =", _fy.shape)
        print("_fxy.shape =", _fxy.shape)
        for i in range(_f.ndim):
            assert _f.shape[i] == len(_coords[i])

    # Plotting
    # -------------------------------------------------------------------------
    if plot_kws_marginal_only is None:
        plot_kws_marginal_only = dict()
    for key in plot_kws:
        plot_kws_marginal_only.setdefault(key, plot_kws[key])
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("figwidth", ncols * (8.5 / 13.0))
    fig_kws.setdefault("share", False)
    fig_kws.setdefault("xticks", [])
    fig_kws.setdefault("yticks", [])
    fig_kws.setdefault("xspineloc", "neither")
    fig_kws.setdefault("yspineloc", "neither")

    # Create the figure.
    hspace = nrows * [space]
    wspace = ncols * [space]
    if plot_marginals:
        hspace[-1] = wspace[-1] = gap
    else:
        hspace = hspace[:-1]
        wspace = wspace[:-1]
    fig, axes = pplt.subplots(
        ncols=(ncols + 1 if plot_marginals else ncols),
        nrows=(nrows + 1 if plot_marginals else nrows),
        hspace=hspace,
        wspace=wspace,
        **fig_kws,
    )

    # Plot the projections:
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[nrows - 1 - i, j]
            idx = psi.make_slice(_f.ndim, axis=axis_slice, ind=[(j, j + 1), (i, i + 1)])
            ax = image(
                psi.project(_f[idx], axis_view),
                x=_coords[axis_view[0]],
                y=_coords[axis_view[1]],
                ax=ax,
                **plot_kws,
            )
    if plot_marginals:
        for i, ax in enumerate(reversed(axes[:-1, -1])):
            ax = image(
                _fy[:, :, i],
                x=_coords[axis_view[0]],
                y=_coords[axis_view[1]],
                ax=ax,
                **plot_kws_marginal_only,
            )
        for i, ax in enumerate(axes[-1, :-1]):
            ax = image(
                _fx[:, :, i],
                x=_coords[axis_view[0]],
                y=_coords[axis_view[1]],
                ax=ax,
                **plot_kws_marginal_only,
            )
        ax = image(
            _fxy,
            x=_coords[axis_view[0]],
            y=_coords[axis_view[1]],
            ax=axes[-1, -1],
            **plot_kws_marginal_only,
        )

    # Add labels:
    if annotate and dims is not None:
        # Label the view dimensions.
        annotate_kws = dict(
            color="white",
            xycoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="center",
        )
        for i, xy in enumerate([(0.5, 0.13), (0.12, 0.5)]):
            axes[0, 0].annotate(_dims[axis_view[i]], xy=xy, **annotate_kws)

        # Label the slice dimensions. Print dimension labels with arrows like this:
        # "<----- x ----->" on the bottom and right side of the main panel.
        arrow_length = 2.5  # arrow length
        text_length = 0.15  # controls space between dimension label and start of arrow
        annotate_kws = dict(
            xycoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ilast = -2 if plot_marginals else -1  # index of last ax in main panel
        anchors = (axes[ilast, ncols // 2], axes[nrows // 2, ilast])
        anchors[0].annotate(
            _dims[axis_slice[0]], xy=(0.5, -label_height), **annotate_kws
        )
        anchors[1].annotate(
            _dims[axis_slice[1]], xy=(1.0 + label_height, 0.5), **annotate_kws
        )
        annotate_kws["arrowprops"] = dict(arrowstyle="->", color="black")
        for arrow_direction in (1.0, -1.0):
            anchors[0].annotate(
                "",
                xy=(0.5 + arrow_direction * arrow_length, -label_height),
                xytext=(0.5 + arrow_direction * text_length, -label_height),
                **annotate_kws,
            )
            anchors[1].annotate(
                "",
                xy=(1.0 + label_height, 0.5 + arrow_direction * arrow_length),
                xytext=(1.0 + label_height, 0.5 + arrow_direction * text_length),
                **annotate_kws,
            )
    if return_indices:
        return axes, ind_slice
    return axes


def interactive_proj2d(
    f,
    coords=None,
    default_ind=(0, 1),
    slice_type="int",  # {'int', 'range'}
    dims=None,
    units=None,
    prof_kws=None,
    cmaps=None,
    frac_thresh=None,
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
    prof_kws : dict
        Key word arguments for 1D profile plots.
    cmaps : list
        Color map options for dropdown menu.

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
    if prof_kws is None:
        prof_kws = dict()
    prof_kws.setdefault("lw", 1.0)
    prof_kws.setdefault("alpha", 0.5)
    prof_kws.setdefault("color", "white")
    prof_kws.setdefault("scale", 0.14)
    if cmaps is None:
        cmaps = ["viridis", "dusk_r", "mono_r", "plasma"]
    plot_kws.setdefault("colorbar", True)
    plot_kws["prof_kws"] = prof_kws
    plot_kws["thresh_type"] = "frac"

    # Widgets
    cmap = widgets.Dropdown(options=cmaps, description="cmap")
    thresh_checkbox = widgets.Checkbox(value=True, description="thresh")
    thresh = widgets.FloatSlider(
        value=-3.3,
        min=-8.0,
        max=0.0,
        step=0.1,
        description="thresh (log)",
        continuous_update=True,
    )
    discrete = widgets.Checkbox(value=False, description="discrete")
    log = widgets.Checkbox(value=False, description="log")
    contour = widgets.Checkbox(value=False, description="contour")
    profiles = widgets.Checkbox(value=False, description="profiles")
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
        # Hide other sliders based on checkboxes.
        thresh.layout.display = None if thresh_checkbox.value else "none"

    # Update the slider list automatically.
    for element in (dim1, dim2, *checks, thresh_checkbox):
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
        idx = psi.make_slice(f.ndim, axis_slice, ind)
        _f = psi.project(f[idx], axis_view)
        # Update plotting key word arguments.
        plot_kws["cmap"] = kws["cmap"]
        plot_kws["fill_value"] = 0
        plot_kws["norm"] = "log" if kws["log"] else None
        plot_kws["profx"] = plot_kws["profy"] = kws["profiles"]
        plot_kws["thresh"] = (10.0 ** kws["thresh"]) if kws["thresh_checkbox"] else None
        # Plot the projection onto the specified axes.
        fig, ax = pplt.subplots()
        ax = image(_f, x=coords[axis_view[0]], y=coords[axis_view[1]], ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()

    # Pass key word arguments for `update`.
    kws = {
        "cmap": cmap,
        "log": log,
        "profiles": profiles,
        "thresh_checkbox": thresh_checkbox,
        "thresh": thresh,
        "dim1": dim1,
        "dim2": dim2,
    }
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)


def interactive_proj1d(
    f,
    coords=None,
    default_ind=0,
    slice_type="int",  # {'int', 'range'}
    dims=None,
    units=None,
    kind="bar",
    fig_kws=None,
    **plot_kws,
):
    """2D partial projection of image interactive slicing.

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
    plot_kws.setdefault("color", "black")
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("figsize", (4.5, 1.5))

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
        idx = psi.make_slice(f.ndim, axis_slice, ind)
        profile = psi.project(f[idx], axis_view)
        if np.max(profile) > 0:
            profile = profile / np.sum(profile)
        # Plot the projection.
        fig, ax = pplt.subplots(**fig_kws)
        ax.format(xlabel=dims_units[axis_view])
        if kind == "bar":
            ax.bar(coords[axis_view], profile, **plot_kws)
        elif kind == "line":
            ax.plot(coords[axis_view], profile, **plot_kws)
        elif kind == "step":
            ax.plot(coords[axis_view], profile, drawstyle="steps-mid", **plot_kws)
        plt.show()

    kws = {"dim1": dim1}
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)