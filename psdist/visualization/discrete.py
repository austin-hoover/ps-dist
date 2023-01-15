"""Plotting routines for discrete sets of points in 2n-dimensional phase space."""
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt

from .. import image as _image
from .. import discrete as _discrete
from ..utils import centers_from_edges

from . import visualization as vis
from . import image as vis_image


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


def plot_rms_ellipse(
    X, ax=None, level=1.0, center_at_mean=True, **ellipse_kws
):
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


def scatter(X, ax=None, **kws):
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
    return ax.scatter(X[:, 0], X[:, 1], **kws)


def hist(X, bins='auto', limits=None, ax=None, **kws):
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
    f, coords = _discrete.histogram(X, bins=bins, limits=limits, centers=True)
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
    estimator = _discrete.gaussian_kde(X, **kde_kws)
    density = estimator.evaluate(_image.get_grid_coords(*coords).T)
    density = np.reshape(density, [len(c) for c in coords])
    return vis_image.plot2d(density, coords=coords, ax=ax, **kws)


def plot2d(
    X, 
    kind='hist',
    rms_ellipse=False, 
    rms_ellipse_kws=None,
    ax=None,
    **kws
):
    if kind == 'hist':
        kws.setdefault('mask', True)
    func = None
    if kind in ['hist', 'contour', 'contourf']:
        func = hist
        if kind in ['contour', 'contourf']:
            kws['kind'] = kind
    elif kind == 'scatter':
        func = scatter
    elif kind == 'kde':
        func = kde
    else:
        raise ValueError('Invalid plot kind.')
    _out = func(X, ax=ax, **kws)
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_rms_ellipse(X, ax=ax, **rms_ellipse_kws)
    return _out
    
    
def jointplot():
    # This will be like seaborn.jointplot (top/right panel axs).
    raise NotImplementedError
    
    
def corner(
    X,
    diag=True,
    limits=None,
    labels=None,
    diag_height_frac=0.6,
    autolim_kws=None,
    diag_kws=None,
    fig_kws=None,
    return_fig=False,
    return_mesh=False,
    prof_edge_only=False,
    axs=None,
    modify_limits=True,
    **kws,
):
    """Plot one- and two-dimensional projections in a corner plot.

    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinates of k points in n-dimensional space.
    diag : bool
        Whether to include the diagonal subplots in the figure. If False, we get
        an n-1 x n-1 matrix of subplots instead of an n x n matrix..
    limits : list[tuple]
        The (min, max) coordinates for each dimension. This is used to set the
        axis limits, as well as for data binning if plotting a histogram.
    labels : list[str]
        The axis labels.
    diag_height_frac : float
        Reduce the height of 1D profiles (diagonal subplots) relative to the
        y axis height.
    autolim_kws : dict
        Key word arguments passed to `autolim`.
    diag_kws : dict
        Key word arguments passed to 1D plotting function on diagonal axes.
    fig_kws : dict
        Key word arguments passed to `pplt.subplots` such as 'figwidth'.
        Whether to return `fig` in addition to `axes`.
    return_mesh : bool
        Whether to also return a mesh from one of the pcolor plots. This is
        useful if you want to put a colorbar on the figure later.
    prof_edge_only : bool
        If 'profx' and 'profy' are in `kws`, only plot the x profiles on the
        bottom row and y profiles on the left column of subplots.
    axs : proplot.gridspec
        If provided, plot on these axes instead of creating new ones.
    **plot_kws
        Key word arguments passed to `plot2d`.

    Returns
    -------
    axes : proplot.gridspec
        Array of subplot axes.
    Optional:
        fig : proplot.figure
            Proplot figure object.
        mesh : matplotlib.collections.QuadMesh
            Mesh from the latest call of `pcolormesh`. This is helpful if you want a
            global colorbar.
    """
    n = X.shape[1]
    if diag_kws is None:
        diag_kws = dict()
        
    diag_kws.setdefault('color', 'black')
    diag_kws.setdefault('lw', 1.0)
    diag_kws.setdefault('kind', 'step')
    kws.setdefault('kind', 'hist')

    if limits is None:
        if autolim_kws is None:
            autolim_kws = dict()
        limits = auto_limits(X, **autolim_kws)
    if axs is None:
        if fig_kws is None:
            fig_kws = dict()
        fig, axs = vis._setup_corner(n, diag, labels, limits, **fig_kws)
    else:
        return_fig = False
        if modify_limits: 
            if axs.shape[1] == n:
                old_limits = [axs[i, i].get_xlim() for i in range(n)]
            else:
                old_limits = [axs[-1, i].get_xlim() for i in range(n - 1)] + [axs[-1, 0].get_ylim()]
            limits = [
                (min(old_lim[0], lim[0]), max(old_lim[1], lim[1]))
                for old_lim, lim in zip(old_limits, limits)
            ]
            vis._set_corner_limits(axs, limits, diag=(axs.shape[1] == n))

    # Univariate plots. Remember histogram bins and use them for 2D histograms.
    bins = 'auto'
    if 'bins' in kws:
        bins = kws.pop('bins')
    edges, centers = [], []
    for i in range(n):
        heights, _edges = np.histogram(X[:, i], bins, limits[i], density=True)
        heights = heights / np.max(heights)
        _centers = centers_from_edges(_edges)
        edges.append(_edges)
        centers.append(_centers)
        if diag:
            vis.plot1d(_centers, heights, ax=axs[i, i], **diag_kws)

    # Bivariate plots:
    for i, row in enumerate(range(int(diag), axs.shape[0]), start=1):
        for j, col in enumerate(range(i)):
            ax = axs[row, col]
            axis = (j, i)
            
            # If an image plot, determine if we should 
            # plot the 1D projections.
            if prof_edge_only and ('profx' in kws or 'profy' in kws):
                kws['profx'] = row == axs.shape[0] - 1
                kws['profy'] = col == 0
                
            # If plotting a histogram, use the bin edges computed in the
            # univariate histograms above.
            if kws['kind'] == 'hist':
                kws['bins'] = [edges[j], edges[i]]

            plot2d(X[:, axis], ax=ax, **kws)

    # Modify diagonal y axis limits.
    if diag:
        for i in range(n):
            axs[i, i].set_ylim(0, 1.0 / diag_height_frac)
    # Return items
    if return_fig:
        if return_mesh:
            return fig, axs, mesh
        return fig, axs
    return axs
    
    
def slice_matrix():
    # Slice matrix plot for bunch.
    raise NotImplementedError
    
    
def plot2d_interactive_slice(
    X,
    limits=None,
    nbins=30,
    default_ind=(0, 1),
    slice_type='int',
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
        Key word arguments passed to `plot2d`.
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
        
    # Set default plot arguments.
    plot_kws.setdefault('kind', 'hist')
    plot_kws.setdefault('colorbar', True)
    plot_kws['prof_kws'] = prof_kws

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
        dim1 = kws['dim1']
        dim2 = kws['dim2']
        nbins = kws['nbins']
        nbins_plot = kws['nbins_plot']
        autobin = kws['autobin']
        ind, checks = [], []
        for i in range(100):
            if f'check{i}' in kws:
                checks.append(kws[f'check{i}'])
            if f'slider{i}' in kws:
                _ind = kws[f'slider{i}']
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
            _X = _discrete.slice_planar(X, axis=axis_slice, center=center, width=width)
        else:
            _X = X[:, :]

        # Update plotting key word arguments.
        plot_kws['bins'] = 'auto' if autobin else nbins_plot
        plot_kws['limits'] = limits
        plot_kws['norm'] = 'log' if kws['log'] else None
        
        # Plot the selecting points.
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