"""Plotting routines for phase space bunches."""
import numpy as np

from .. import bunch as _bunch

from . import visualization as vis
from . import image as _vis_image


def auto_limits(X, sigma=None, pad=0.0, zero_center=False, share_xy=False):
    """Determine axis limits from coordinate array.

    Parameters
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
    maxs = mins + padding
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


def plot_rms_ellipse(
    X, axis=(0, 1), ax=None, level=1.0, center_at_mean=True, **ellipse_kws
):
    """Compute and plot RMS ellipse from bunch coordinates.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    axis : 2-tuple of int
        The dimensions to plot.
    ax : Axes
        The axis on which to plot.
    level : number of list of numbers
        If a number, plot the rms ellipse inflated by the number. If a list
        of numbers, repeat for each number.
    center_at_mean : bool
        Whether to center the ellipse at the image centroid.
    """
    center = np.mean(X[:, axis], axis=0) 
    if center_at_mean:
        center = np.zeros(2)
    Sigma = np.cov(X[:, axis].T)
    return vis.rms_ellipse(Sigma, center, level=level, ax=ax, **ellipse_kws)


def scatter(X, axis=(0, 1), ax=None, **kws):
    """Convenience function for 2D scatter plot.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    axis : 2-tuple of int
        The dimensions to plot.
    ax : Axes
        The axis on which to plot.
    **kws
        Key word arguments passed to `ax.scatter`.
    """
    kws.setdefault('c', 'black')
    kws.setdefault('s', 1.0)
    return ax.scatter(X[:, axis[0]], X[:, axis[1]], **kws)


def hist(X, axis=(0, 1), bins='auto', binrange=None, ax=None, **kws):
    """Convenience function for 2D histogram with auto-binning.

    For more options, I recommend seaborn.histplot.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    axis : 2-tuple of int
        The dimensions to plot.
    ax : Axes
        The axis on which to plot.
    limits, bins : see `psdist.bunch.histogram`.
    **kws 
        Key word arguments passed to `plotting.image`.
    """
    f, coords = _bunch.histogram(X[:, axis], bins=bins, binrange=binrange, centers=True)
    return _vis_image.plot2d(f, coords=coords, ax=ax, **kws)
    
        
def kde(X, axis=(0, 1), ax=None, kde_kws=None, **kws):
    """Plot kernel density estimation (KDE).
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    axis : 2-tuple of int
        The dimensions to plot.
    ax : Axes
        The axis on which to plot.
    kde_kws : dict
        Key word arguments passed to `psdist.bunch.kde`.
    **kws
        Key word arguments passed to `psdist.visualization.image.plot2`.
    """
    if kde_kws is None:
        kde_kws = dict()
    kde_kws.setdefault('axis', (0, 1))
    density, coords = _bunch.kde(X[:, axis], **kde_kws)
    return _vis_image.plot2d(density, coords=coords, ax=ax, **kws)


def plot2d(
    X, 
    axis=(0, 1), 
    kind='hist',
    rms_ellipse=False, 
    rms_ellipse_kws=None,
    ax=None,
    **kws
):
    if kind in ['hist', 'kde']:
        kws.setdefault('process_kws', dict(mask_zero=True))
    func = None
    if kind == 'hist':
        func = hist
    elif kind == 'scatter':
        func = scatter
    elif kind == 'kde':
        func = kde
    else:
        raise ValueError('Invalid plot kind.')
    func(X, axis=axis, ax=ax, **kws)
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_rms_ellipse(axis=axis, ax=ax, **rms_ellipse_kws)
    
    
def jointplot():
    # This will be like seaborn.jointplot (top/right panel axes).
    raise NotImplementedError
    
    
def corner():
    # Corner plot for bunch.
    raise NotImplementedError
    
    
def slice_matrix():
    # Slice matrix plot for bunch.
    raise NotImplementedError
    
    
def interactive_proj2d(
    X,
    limits=None,
    nbins=30,
    default_ind=(0, 1),
    slice_type="int",  # {'int', 'range'}
    dims=None,
    units=None,
    prof_kws=None,
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
        Key word arguments passed to `image`.
    """
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
    plot_kws.setdefault("colorbar", True)
    plot_kws["prof_kws"] = prof_kws

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
    autobin = widgets.Checkbox(description="auto plot res", value=True)
    log = widgets.Checkbox(description="log", value=False)
    prof = widgets.Checkbox(description="profiles", value=False)
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

        # Return nothing if input does not make sense.
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
            for _axis in axis_slice:
                _edges = edges[_axis]
                imin, imax = ind[_axis]
                width.append(_edges[imax] - _edges[imin])
                center.append(0.5 * (_edges[imax] + _edges[imin]))
            Xs = psb.slice_box(X, axis=axis_slice, center=center, width=width)
        else:
            Xs = X[:, :]

        # Compute 2D histogram of remaining particles.
        _nbins = "auto" if autobin else nbins_plot
        xedges = np.histogram_bin_edges(
            Xs[:, axis_view[0]], bins=_nbins, range=limits[axis_view[0]]
        )
        yedges = np.histogram_bin_edges(
            Xs[:, axis_view[1]], bins=_nbins, range=limits[axis_view[1]]
        )
        edges = [xedges, yedges]
        centers = [utils.centers_from_edges(e) for e in edges]
        f, _, _ = np.histogram2d(
            Xs[:, axis_view[0]], Xs[:, axis_view[1]], bins=edges
        )

        # Update plot key word arguments.
        plot_kws["norm"] = "log" if kws["log"] else None
        plot_kws["profx"] = plot_kws["profy"] = kws["prof"]
        # Plot the image.
        fig, ax = pplt.subplots()
        ax = image(f, x=centers[0], y=centers[1], ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()

    kws = dict()
    kws["log"] = log
    kws["prof"] = prof
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