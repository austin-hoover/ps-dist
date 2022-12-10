"""Plotting routines for phase space distributions."""
from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt
from scipy import optimize as opt

from . import bunch as psb
from . import image as psi
from . import utils


def linear_fit(x, y):
    """Return (yfit, slope, intercept) from linear fit."""
    def fit(x, slope, intercept):
        return slope * x + intercept
    
    popt, pcov = opt.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def process_limits(mins, maxs, pad=0.0, zero_center=False):
    """Make all position coordinates have the same limits (same for momentum coordinates).
    
    Parameters
    ----------
    mins, maxs : list
        The minimum and maximum coordinates for each dimension (x, x', y, y', z, z').
    pad : float
        Fractional padding to apply to the limits.
    zero_center : bool
        Whether to center the limits on zero.
    
    Returns
    -------
    mins, maxs : list
        The new limits.
    """
    # Same limits for x/y and x'/y'
    widths = np.abs(mins - maxs)
    for (i, j) in [[0, 2], [1, 3]]:
        delta = 0.5 * (widths[i] - widths[j])
        if delta < 0.0:
            mins[i] -= abs(delta)
            maxs[i] += abs(delta)
        elif delta > 0.0:
            mins[j] -= abs(delta)
            maxs[j] += abs(delta)
    # Pad the limits by fractional amount `pad`.
    deltas = 0.5 * np.abs(maxs - mins)
    padding = deltas * pad
    mins -= padding
    maxs += padding
    if zero_center:
        maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
        mins = -maxs
    return mins, maxs


def auto_limits(X, sigma=None, **kws):
    """Determine axis limits from coordinate array.
    
    Parameters
    ----------
    X : ndarray, shape (k, n)
        Coordinate array for k points in n-dimensional space.
    sigma : float
        If a number is provided, it is used to set the limits relative to 
        the standard deviation of the distribution.
    **kws
        Key word arguments for `process_limits`.
        
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
    mins, maxs = process_limits(mins, maxs, **kws)
    return [(lo, hi) for lo, hi in zip(mins, maxs)]


# Images
# ------------------------------------------------------------------------------
def plot1d(x, y, ax=None, flipxy=False, kind="step", **kws):
    """Convenience function for one-dimensional line/step/bar plots."""
    funcs = {
        "line": ax.plot,
        "bar": ax.bar,
        "step": ax.plot,
    }
    if kind == "step":
        kws.setdefault("drawstyle", "steps-mid")
    if flipxy:
        x, y = y, x
        funcs["bar"] = ax.barh
    return funcs[kind](x, y, **kws)


def plot_profile(
    f,
    xcoords=None,
    ycoords=None,
    ax=None,
    profx=True,
    profy=True,
    kind='step',
    scale=0.12,
    **plot_kws,
):
    """Overlay a 1D projection on top of a 2D image.
    
    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    xcoords, ycoords : list
        Coordinates of pixel centers.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    profx, profy : bool
        Whether to plot the x/y profile.
    kind : {'step', 'bar', 'line'}
        The type of 1D plot.
    scale : float
        Maximum of the 1D plot relative to the axes limits.
    **plot_kws
        Key word arguments for the 1D plotting function.
    """
    if xcoords is None:
        xcoords = np.arange(f.shape[1])
    if ycoords is None:
        ycoords = np.arange(f.shape[0])
    plot_kws.setdefault("lw", 0.75)
    plot_kws.setdefault("color", "white")

    def _normalize(profile):
        pmax = np.max(profile)
        if pmax > 0:
            profile = profile / pmax
        return profile
    
    px, py = [_normalize(np.sum(f, axis=i)) for i in (1, 0)]
    yy = ycoords[0] + scale * np.abs(ycoords[-1] - ycoords[0]) * px 
    xx = xcoords[0] + scale * np.abs(xcoords[-1] - xcoords[0]) * py
    yy -= (np.min(yy) - ycoords[0])
    xx -= (np.min(xx) - xcoords[0])
    for i, (x, y) in enumerate(zip([xcoords, ycoords], [yy, xx])):
        if i == 0 and not profx:
            continue
        if i == 1 and not profy:
            continue
        plot1d(x, y, ax=ax, flipxy=i, kind=kind, **plot_kws)
    return ax


def plot_image(
    f,
    x=None,
    y=None,
    ax=None,
    profx=False,
    profy=False,
    prof_kws=None,
    thresh=None,
    thresh_type='abs',
    contour=False,
    contour_kws=None,
    return_mesh=False,
    fill_value=None,
    mask_zero=False,
    floor=None,
    **plot_kws,
):
    """Plot a 2D image.
    
    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    x, y : list
        Coordinates of pixel centers.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    profx, profy : bool
        Whether to plot the x/y profile.
    prof_kws : dict
        Key words arguments for `plot_profile`.
    thresh : float
        Set elements below this value to zero.
    thresh_type : {'abs', 'frac'}
        If 'frac', `thresh` is a fraction of the maximum element in `f`. 
    'contour' : bool
        Whether to plot a contour on top of the image.
    contour_kws : dict
        Key word arguments for `ax.contour`.
    return_mesh : bool
        Whether to return a mesh from `ax.pcolormesh`.
    fill_value : float
        If not None, fills in masked values of `f`.
    mask_zero : bool
        Whether to mask zero values of `f`.
    floor : float
        Add `floor * min(f[f > 0])` to `f`.
    **plot_kws
        Key word arguments for `ax.pcolormesh`.
    """
    plot_kws.setdefault("ec", "None")
    plot_kws.setdefault("linewidth", 0.0)
    plot_kws.setdefault("rasterized", True)
    log = "norm" in plot_kws and plot_kws["norm"] == "log"
    
    f = f.copy()
    if fill_value is not None:
        f = np.ma.filled(f, fill_value=fill_value)
    if thresh is not None:
        if thresh_type == "frac":
            thresh = thresh * np.max(f)
        f[f < max(1.0e-12, thresh)] = 0
    if mask_zero:
        f = np.ma.masked_less_equal(f, 0)
    if floor is not None:
        _floor = 1.0e-12
        if np.max(f) > 0.0:
            f_min_pos = np.min(f[f > 0])
            floor = floor * f_min_pos
        f = f + floor
    if log:
        if np.any(f == 0):
            f = np.ma.masked_less_equal(f, 0)
        if "colorbar" in plot_kws and plot_kws["colorbar"]:
            if "colorbar_kw" not in plot_kws:
                plot_kws["colorbar_kw"] = dict()
            plot_kws["colorbar_kw"]["formatter"] = "log"
    if contour_kws is None:
        contour_kws = dict()
    contour_kws.setdefault("color", "white")
    contour_kws.setdefault("lw", 1.0)
    contour_kws.setdefault("alpha", 0.5)
    if prof_kws is None:
        prof_kws = dict()
    if x is None:
        x = np.arange(f.shape[0])
    if y is None:
        y = np.arange(f.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    mesh = ax.pcolormesh(x, y, f.T, **plot_kws)
    if contour:
        ax.contour(x, y, f.T, **contour_kws)
    if profx or profy:
        plot_profile(
            f, xcoords=x, ycoords=y, ax=ax, profx=profx, profy=profy, **prof_kws
        )
    if return_mesh:
        return ax, mesh
    else:
        return ax


def _setup_corner(n, diag, labels, limits=None, **fig_kws):
    """Set up corner plot axes."""
    if labels is None:
        labels = n * [""]
    nrows = ncols = n if diag else n - 1
    start = 1 if diag else 0
    fig_kws.setdefault("figwidth", 1.5 * nrows)
    fig_kws.setdefault("aligny", True)
    fig, axes = pplt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=1,
        sharey=1,
        spanx=False,
        spany=False,
        **fig_kws,
    )
    for i in range(nrows):
        for j in range(ncols):
            if j > i:
                axes[i, j].axis("off")
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[(1 if diag else 0):, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(nrows):
        axes[:-1, i].format(xticklabels=[])
        axes[i, 1:].format(yticklabels=[])
    for ax in axes:
        ax.format(xspineloc="bottom", yspineloc="left")
    if diag:
        for i in range(n):
            axes[i, i].format(yspineloc="neither")
    if limits is not None:
        for j in range(ncols):
            axes[:, j].format(xlim=limits[j])
        _limits = limits if diag else limits[1:]
        for i in range(start, nrows):
            axes[i, :].format(ylim=_limits[i])
    axes.format(xtickminor=True, ytickminor=True, xlocator=('maxn', 3), ylocator=('maxn', 3))
    return fig, axes


def corner(
    data,
    kind="hist",
    diag_kind="step",
    coords=None,
    limits=None,
    labels=None,
    samples=None,
    diag_height_frac=0.6,
    autolim_kws=None,
    diag_kws=None,
    fig_kws=None,
    prof=False,
    prof_kws=None,
    return_fig=False,
    return_mesh=False,
    **plot_kws,
):
    """Plot one- and two-dimensional projections in a corner plot.

    Parameters
    ----------
    data : ndarray
        If `data.ndim == 2`, we have the coordinates of k points in n-dimensional 
        space; otherwise, we have an n-dimensional image.
    kind : {'hist', 'scatter'}
        The kind of 2D plot to make if `data` is a list of points. 
            'hist': 2D histogram
            'scatter': 2D scatter plot
    diag_kind : {'line', 'step', 'bar', 'None'}
        Kind of 1D plot on diagonal axes. Any variant of 'None', 'none', None 
        will remove the diagonal axes from the figure, resulting in a D-1 x D-1
        array of subplots instead of a D x D array.
    coords : list[ndarray]
        Coordinates along each axis of the grid (if `data` is an image).
    limits : list[tuple]
        The (min, max) coordinates for each dimension. This is used to set the
        axis limits, as well as for data binning if plotting a histogram.
    labels : list[str]
        The axis labels.
    samples : int or float
        Number of samples to use in scatter plots. If less than 1, specifies
        the fraction of points.
    diag_height_frac : float
        Reduce the height of 1D profiles (diagonal subplots) relative to the 
        y axis height.
    autolim_kws : dict
        Key word arguments passed to `autolim`.
    diag_kws : dict
        Key word arguments passed to 1D plotting function on diagonal axes.
    fig_kws : dict
        Key word arguments passed to `pplt.subplots` such as 'figwidth'.
    prof : bool or 'edges'
        Whether to overlay 1D profiles on 2D plots in off-diagonal axes. If
        'edges', only plot profiles in the left column and bottom row of
        the figure. This is a good option if not using diagonal subplots.
    prof_kws : dict
        Key word arguments passed to `plot_profiles`.
    return_fig : bool
        Whether to return `fig` in addition to `axes`.
    return_mesh : bool
        Whether to also return a mesh from one of the pcolor plots. This is
        useful if you want to put a colorbar on the figure later.
    **plot_kws
        Key word arguments passed to 2D plotting function.
    
    Returns
    -------
    fig : proplot.figure
        Proplot figure object.
    axes : proplot.gridspec
        Array of subplot axes.
    """
    # Determine whether data is point cloud or image.
    n = data.ndim
    pts = False
    if n == 2:
        n = data.shape[1]
        pts = True
        
    # Parse arguments
    diag = diag_kind in ["line", "bar", "step"]
    start = 1 if diag else 0
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault("color", "black")
    diag_kws.setdefault("lw", 1.0)
    if pts and kind == "scatter":
        plot_kws.setdefault("s", 6)
        plot_kws.setdefault("c", "black")
        plot_kws.setdefault("marker", ".")
        plot_kws.setdefault("ec", "none")
        if "color" in plot_kws:
            plot_kws["c"] = plot_kws.pop("color")
        if "ms" in plot_kws:
            plot_kws["s"] = plot_kws.pop("ms")
    elif (pts and kind == "hist") or not pts:
        plot_kws.setdefault("ec", "None")

    # Create the figure.
    if (not pts) and (coords is None):
        coords = [np.arange(s) for s in data.shape]
    if autolim_kws is None:
        autolim_kws = dict()
    if fig_kws is None:
        fig_kws = dict()
    if limits is None:
        if pts:
            limits = auto_limits(data, **autolim_kws)
        else:
            limits = [(c[0], c[-1]) for c in coords]
    fig, axes = _setup_corner(n, diag, labels, limits, **fig_kws)

    # Discrete points
    if pts:
        # Univariate plots
        bins = "auto"
        if "bins" in plot_kws:
            bins = plot_kws.pop("bins")
        edges, centers = [], []
        for i in range(n):
            heights, _edges = np.histogram(data[:, i], bins, limits[i], density=True)
            heights = heights / np.max(heights)
            _centers = utils.bin_centers(_edges)
            edges.append(_edges)
            centers.append(_centers)
            if diag:
                plot1d(_centers, heights, ax=axes[i, i], kind=diag_kind, **diag_kws)

        # Take random sample of points.
        idx = np.arange(data.shape[0])
        if samples is not None and samples < data.shape[0]:
            if samples < 1:
                # Convert from fraction of points to number of points.
                samples = samples * data.shape[0]
            idx = utils.rand_rows(idx, int(samples))

        # Bivariate plots
        for ii, i in enumerate(range(start, axes.shape[0])):
            for j in range(ii + 1):
                ax = axes[i, j]
                if kind == "scatter":
                    ax.scatter(data[idx, j], data[idx, ii + 1], **plot_kws)
                elif kind == "hist":
                    _im, _, _ = np.histogram2d(
                        data[:, j], data[:, ii + 1], (edges[j], edges[ii + 1])
                    )
                    if prof == "edges":
                        profy = j == 0
                        profx = i == axes.shape[0] - 1
                    else:
                        profx = profy = prof
                    plot_image(
                        _im,
                        x=centers[j],
                        y=centers[ii + 1],
                        ax=ax,
                        profx=profx,
                        profy=profy,
                        prof_kws=prof_kws,
                        **plot_kws,
                    )
    # Multidimensional image
    else:
        # Bivariate plots
        for ii, i in enumerate(range(start, axes.shape[0])):
            for j in range(ii + 1):
                ax = axes[i, j]
                if prof == "edges":
                    profy = j == 0
                    profx = i == axes.shape[0] - 1
                else:
                    profx = profy = prof
                image = psi.project(data, (j, ii + 1))
                image = image / np.max(image)
                ax, mesh = plot_image(
                    image,
                    x=coords[j],
                    y=coords[ii + 1],
                    ax=ax,
                    profx=profx,
                    profy=profy,
                    prof_kws=prof_kws,
                    return_mesh=True,
                    **plot_kws,
                )
        # Univariate plots
        if diag:
            for i in range(n):
                profile = psi.project(data, i)
                profile = profile / np.max(profile)
                if 'fill_value' in plot_kws:
                    profile = np.ma.filled(profile, fill_value=plot_kws['fill_value'])
                plot1d(coords[i], profile, ax=axes[i, i], kind=diag_kind, **diag_kws)                
    # Modify diagonal y axis limits.
    if diag:
        for i in range(n):
            axes[i, i].set_ylim(0, 1.0 / diag_height_frac)
    # Return items
    if return_fig:
        if return_mesh:
            return fig, axes, mesh
        return fig, axes
    return axes


def _setup_matrix_slice(nrows=9, ncols=9, space=0.1, gap=2.0, **fig_kws):
    """Set up matrix_slice figure axes."""
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 8.5)
    fig_kws.setdefault('share', False)
    fig_kws.setdefault('xticks', [])
    fig_kws.setdefault('yticks', [])
    fig_kws.setdefault('xspineloc', 'neither')
    fig_kws.setdefault('yspineloc', 'neither')
    hspace = nrows * [space]
    wspace = ncols * [space]
    hspace[-1] = wspace[-1] = gap
    fig, axes = pplt.subplots(
        ncols=ncols+1, nrows=nrows+1, 
        hspace=hspace, wspace=wspace,
        **fig_kws
    )
    return fig, axes


def _annotate_matrix_slice(axes, islice, iview, dims, height=0.2, 
                           length=2.5, textlength=0.15, arrowprops=None):
    """Add labels to the axes of matrix_slice figure."""
    nrows = axes.shape[0] - 1
    ncols = axes.shape[1] - 1
    if arrowprops is None:
        arrowprops = dict()
    arrowprops.setdefault('arrowstyle', '->')
    arrowprops.setdefault('color', 'black')
    annotate_kws = dict(
        xycoords='axes fraction', 
        horizontalalignment='center', 
        verticalalignment='center',
    )
    for sign in (1.0, -1.0):
        ax = axes[-1, ncols // 2]
        _height = height + 1.0
        ax.annotate(
            '', xy=(0.5 + sign * length, _height),
            xytext=(0.5 + sign * textlength, _height),
            arrowprops=arrowprops,
            **annotate_kws
        )
        ax.annotate(dims[islice[0]], xy=(0.5, _height), **annotate_kws)

        ax = axes[nrows // 2, -1]
        _height = -height
        ax.annotate(
            '', xy=(_height, 0.5 + sign * length),
            xytext=(_height, 0.5 + sign * textlength),
            arrowprops=arrowprops,
            **annotate_kws
        )
        ax.annotate(dims[islice[1]], xy=(_height, 0.5), **annotate_kws)

    ax = axes[0, 0]
    ax.annotate(dims[iview[0]], color='white', xy=(0.5, 0.13), xycoords='axes fraction', 
                horizontalalignment='center', verticalalignment='center')
    ax.annotate(dims[iview[1]], color='white', xy=(0.12, 0.5), xycoords='axes fraction', 
                horizontalalignment='center', verticalalignment='center')
    return axes    


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
    plot_kws_marginal_only=None,
    debug=False,
    **plot_kws
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
    plot_kws_marginal_only : dict
        Key word arguments for the lower-left and upper-right panels, which 
        plot the 3D marginal distributions.
    debug : bool
        Whether to print debugging messages.
    annotate : bool
        Whether to label the axes.
    **plot_kws
        Key word arguments for `plot_image`.
        
    Returns
    -------
    axes
    """
    # Setup
    # -------------------------------------------------------------------------
    if f.ndim < 4:
        raise ValueError('f.ndim < 4')
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
        
    if debug:
        print('Slice indices:')
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
        print('_f.shape =', _f.shape)
        print('_fx.shape =', _fx.shape)
        print('_fy.shape =', _fy.shape)
        print('_fxy.shape =', _fxy.shape)
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
        
    fig, axes = _setup_matrix_slice(
        nrows=nrows, ncols=ncols, space=space, gap=gap, **fig_kws
    )
    if dims is not None:
        axes = _annotate_matrix_slice(axes, axis_slice, axis_view, _dims)
    
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[nrows - 1 - i, j]
            idx = psi.make_slice(_f.ndim, axis=axis_slice, ind=[(j, j + 1), (i, i + 1)])
            plot_image(
                psi.project(_f[idx], axis_view),
                x=_coords[axis_view[0]], 
                y=_coords[axis_view[1]],
                ax=ax,
                **plot_kws
            )
    for i, ax in enumerate(reversed(axes[:-1, -1])):
        plot_image(
            _fy[:, :, i],
            x=_coords[axis_view[0]], 
            y=_coords[axis_view[1]],
            ax=ax,
            **plot_kws_marginal_only
        )
    for i, ax in enumerate(axes[-1, :-1]):
        plot_image(
            _fx[:, :, i],
            x=_coords[axis_view[0]], 
            y=_coords[axis_view[1]],
            ax=ax,
            **plot_kws_marginal_only
        )
    plot_image(
        _fxy,
        x=_coords[axis_view[0]], 
        y=_coords[axis_view[1]],
        ax=axes[-1, -1],
        **plot_kws_marginal_only
    )    
    return axes


# Interactive
# ------------------------------------------------------------------------------
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
    """Interactive plot of 2D projection of distribution `f`.
    
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

    # Widgets
    cmap = widgets.Dropdown(options=cmaps, description="cmap")
    thresh_checkbox = widgets.Checkbox(value=True, description='thresh')
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
        dim1, dim2 = kws['dim1'], kws['dim2']
        ind, checks = [], []
        for i in range(100):
            if f'check{i}' in kws:
                checks.append(kws[f'check{i}'])
            if f'slider{i}' in kws:
                ind.append(kws[f'slider{i}'])
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
        plot_kws['cmap'] = kws['cmap']
        plot_kws['fill_value'] = 0
        plot_kws['norm'] = 'log' if kws['log'] else None
        plot_kws['profx'] = plot_kws['profy'] = kws['profiles']
        plot_kws['thresh'] = (10.0**kws['thresh']) if kws['thresh_checkbox'] else None
        # Plot the projection onto the specified axes. 
        fig, ax = pplt.subplots()
        plot_image(
            _f,
            x=coords[axis_view[0]], 
            y=coords[axis_view[1]], 
            ax=ax, 
            **plot_kws
        )
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()
        
    # Pass key word arguments for `update`.
    kws = {
        'cmap': cmap,
        'log': log,
        'profiles': profiles,
        'thresh_checkbox': thresh_checkbox,
        'thresh': thresh,
        'dim1': dim1,
        'dim2': dim2,
    }
    for i, check in enumerate(checks, start=1):
        kws[f'check{i}'] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f'slider{i}'] = slider  
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
    """1D projection of image `f` with interactive slicing.
    
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
    fig_kws.setdefault('figsize', (4.5, 1.5))

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
        dim1 = kws['dim1']
        ind, checks = [], []
        for i in range(100):
            if f'check{i}' in kws:
                checks.append(kws[f'check{i}'])
            if f'slider{i}' in kws:
                ind.append(kws[f'slider{i}'])
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

    kws = {'dim1': dim1}
    for i, check in enumerate(checks, start=1):
        kws[f"check{i}"] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f"slider{i}"] = slider
    return interactive(update, **kws)


def interactive_proj2d_discrete(
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
    """This mirrors `interactive_proj2d` for point clouds.
        
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
        Key word arguments passed to `plot_image`.
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
    nbins = widgets.IntSlider(min=2, max=100, value=nbins_default, description='grid res')
    nbins_plot = widgets.IntSlider(min=2, max=200, value=nbins_default, description='plot res')
    autobin = widgets.Checkbox(description='auto plot res', value=True)
    log = widgets.Checkbox(description='log', value=False)
    prof = widgets.Checkbox(description='profiles', value=False)
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
            Xs = psb.slice_box(X, axis=axis_slice, center=center, width=width)
        else:
            Xs = X[:, :]
            
        # Compute 2D histogram of remaining particles.
        _nbins = 'auto' if autobin else nbins_plot
        xedges = np.histogram_bin_edges(Xs[:, axis_view[0]], bins=_nbins, range=limits[axis_view[0]])
        yedges = np.histogram_bin_edges(Xs[:, axis_view[1]], bins=_nbins, range=limits[axis_view[1]])
        edges = [xedges, yedges]
        centers = [utils.get_centers(e) for e in edges]
        image, _, _ = np.histogram2d(Xs[:, axis_view[0]], Xs[:, axis_view[1]], bins=edges)
        
        # Update plot key word arguments.
        plot_kws['norm'] = 'log' if kws['log'] else None
        plot_kws['profx'] = plot_kws['profy'] = kws['prof']
        # Plot the image.
        fig, ax = pplt.subplots()
        plot_image(image, x=centers[0], y=centers[1], ax=ax, **plot_kws)
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
