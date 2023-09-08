# Change Log


## [0.1.10] — 2023-09-08

* Add distribution generators (Gaussian and Waterbag). 


## [0.1.9] — 2023-07-22

* Improve diagonal axis sharing in `visualization.CornerGrid`.
* Rename function `visualization.plot1d` &rarr; `visualization.plot_projection` and improve the function.
* Add function `visualization.cloud.proj1d_interactive_slice`.
* Rename function `visualization.stack_limits` &rarr; `visualization.combine_limits`.
* Add function `visualization.center_limits`.
* Add mask option to `visualization.cloud.proj2d_interactive_slice`.
* Fix bug in `cloud.sparse_histogram`.
* In `visualization.JointGrid`, change all "panel_" variables to "marg_".


## [0.1.8] — 2023-07-19

* Add options dict to `visualization.cloud.proj2d_interactive_slice` to decide which checkboxes appear.
* Enable multiple sets of clouds to be compared in `visualization.cloud.proj2d_interactive_slice`.
* Enable three levels of axis limit sharing in `visualization.cloud.proj2d_interactive_slice`.
* Allow list of bin edge coordinates in `ps.cloud.histogram_bin_edges` and `ps.cloud.histogram`.
* Add documentation to histogram functions.
* Add `cloud.sparse_histogram`.
* Add `image.sample_sparse_hist` &rarr; `ps.image.sample_hist`.
* Rename `image.sample_grid` &rarr; `ps.image.sample_hist`.


## [0.1.7] — 2023-06-12

* Add default plot_res and slice_res arguments to `visualization.cloud.proj2d_interactive_slice`.
* Remove limits on plot_res and slice_res widgets.
* Fix bug in colorbar tick locator in `visualization.cloud.proj2d_interactive_slice`.
* Fix `cloud.histogram` to allow one-dimensional arrays.
* Add radial histogram for point clouds. Counts are normalized by spherical hypershell volume.

  
## [0.1.6] — 2023-06-07

* Fixed bug in `visualization.cloud.proj2d_interactive_slice` to allow data sets with different numbers of points.


## [0.1.5] — 2023-06-02

* Fixed bug with `visualization.cloud.slice_planar` limits when only one pair was provided.
* Significantly improved ``visualization.cloud.proj2d_interactive_slice`.
    * Pass an arbitrary number of data sets; outputs side-by-side comparision.
    * Slice and view resolution is now adjusted using a textbox.
    * Slice index slider ranges are dynamically adjusted.


## [0.1.4] — 2023-05-24

* Added absolute vs. relative offset to `visualization.image.plot2d`.
* Subtact image offset when overlaying profiles in `visualization.image.plot2d`.
* Fixed bug: `visualization.SliceGrid` gave error when the number of slices requested was the same as the number of indices in the array.
* Change to convention: 'd' is now used to refer to the number of dimensions, and 'n' is used to refer to the number of points or pixels.
* Fixed bug in `visualization.cloud.proj2d_interactive_slice`: limits were updating.


## [0.1.3] — 2023-04-20

* Updated `visualization.CornerGrid`.
    * Added option to plot on off-diagonal subplots.
    * Added option to show right y spines on diagonal subplots.
    * Changed some keyword arguments.
    * Fix bug: `plot_cloud` was not working when `diag=False`.
* Fixed key word argument default in JointGrid.colorbar.
    

## [0.1.2] — 2023-03-06

* Default ec='None' in scatter plots.
* Fixed bug in `visualization.CornerGrid`: in a fresh plot, if the data limits were smaller than the plot limits, the plot limits were not updated.
* Default histogram limits in `visualization.CornerGrid` are computed using plot limits instead of data limits. 
* Can now pass either bin edges or number of bins as the 'bins' parameter in `visualization.CornerGrid.plot_cloud`.


## [0.1.1] — 2023-01-19

* Added 'blur' option to `image.process`.
* Added `visualization.JointGrid`, `visualization.image.joint`, and `visualization.cloud.joint`.
* Fixed typo causing error in `visualization.cloud.corner`.


## [0.1.0] — 2023-01-17

* Renamed `bunch` &rarr;`cloud`.
    * Added `cloud.gaussian_kde`: computes Gaussian kernel density estimation.
    * Added `cloud.project`: projects onto an axis.
    * Added `cloud.slice_contour`: applies contour shell slice.
    * Added `cloud.transform`: applies function to each point.
    * Renamed `cloud.apply` &rarr; `cloud.linear_transformation`.
    * Renamed `cloud.slice_box` &rarr; `cloud.slice_planar`.
    * Fixed `cloud.slice_ellipsoid` and `cloud.slice_sphere`
* Renamed `plotting` &rarr; `visualization`
    * Added `visualization.CornerGrid`: constructs corner plot, can plot n-dimensional images or clouds.
    * Added `visualization.SliceGrid`: constructs slice matrix grid, can plot four-dimensional images or clouds. (Only images right now.)
    * `visualization.cloud`
        * `visualization.cloud.plot2d`: 2D plot; options for scatter, contour, hist, kde
        * `visualization.cloud.plot_rms_ellipse`: compute/plot the 2D RMS ellipse.
        * `visualization.cloud.corner`: corner plot for discrete data with all options of `plot2d`. (Shortcut to `CornerGrid`.)
        * `visualization.cloud.slice_matrix`: slice matrix plot for discrete data with all options of `plot2d`. (Shortcut to `SliceGrid`. Not working yet.)
        * `visualization.image.plot2d_interactive_slice`: 2D projection with interactive slicing.
    * `visualization.image`
        * `visualization.image.plot2d`: 2D image plot; options for pcolor, contour, contourf.
        * `visualization.image.plot_rms_ellipse`: plots RMS ellipse from image.
        * `visualization.image.plot_profiles`: plot 1D image profiles on top of image.
        * `visualization.image.corner`: corner plot for image data with all options of `plot2d`. (Shortcut to `CornerGrid`.)
        * `visualization.image.slice_matrix`: slice matrix plot for image data with all options of `plot2d`. (Shortcut to `SliceGrid`.)
        * `visualization.image.plot1d_interactive_slice`: 1D projection with interactive slicing.
        * `visualization.image.plot2d_interactive_slice`: 2D projection with interactive slicing.
    * `visualization.plot1d`. Added 'linefilled' and 'stepfilled' plot types.
* `image`
    * Renamed `image.make_slice` &rarr; `image.slice_idx`
    * Renamed `image.make_slice_ellipsoid` &rarr; `image.slice_idx_ellipsoid`
    * Renamed `image.make_slice_contour` &rarr; `image.slice_idx_contour`
    * Added `image._slice`, `image._slice_ellipsoid`, `image._slice_contour`. These return a modified array instead of a slice index. 
    * Improved `image.sample_grid`: ignore bins with zero probability.
* Made some widgets optional in interactive plots.
* Standard imports are now `import psdist as ps`, `import psdist.visualization as psv`, `import psdist.image as psi`, or `import psdist.cloud as psc`.
* Updated example notebooks.


## [0.0.11] - 2023-01-09

* Added `image.sample_grid`: draws random samples from n-dimensional histogram.
* Renamed `plotting.plot1d` &rarr; `plotting.lineplot`


## [0.0.10] - 2022-12-21

* Added version numbers to packages in requirements.txt
* `bunch.decorrelate` now works with 2n-dimensional bunches instead of only six-dimensional bunches.


## [0.0.9] - 2022-12-15

* Added 'return_indices' parameter to `plotting.slice_matrix` to return slice indices.
* Updated examples
* `utils.random_selection` can now take lists as input.
* Fixed typo in `plotting.corner`. There was a lingering variable named 'image', which produced an error after changing the function `plot_image` to `image`. 


## [0.0.8] - 2022-12-15

* `plotting.image_profiles` was flipping axis limits sometimes... added temporary fix.


## [0.0.7] - 2022-12-15

* Added `bunch.downsample`: removes a random selection of points.
* Added `image.enclosing_ellipsoid`: scales covariance ellipsoid until it contains some fraction of points.
* Added `image.enclosing_sphere`: scales sphere until it contains some fraction of points.
* Added `plotting.ellipse`: convenience function to plot an ellipse.
* Added `plotting.rms_ellipse`: plots rms ellipse at various levels given covariance matrix.
* Added `plotting.image_rms_ellipse`: computes rms ellipse from image and plots it.
* Added 'divide_by_max' parameter to `plotting.image`
* Added examples notebook.
* `image.cov is now much faster for high-dimensional images.
* Threshold slider in `plotting.interactive_proj2d` now applies a fractional threshold instead of an absolute threshold.
* `plotting.corner` can now act on existing axes (allowing more than one data set to be plotted).
* Fixed bug when k=None in `utils.random_selection`.
* Fixed naming conflict in `plotting.interactive_proj2d_discrete`.
* Removed 'contour' parameter from `plotting.plot_image`.
* Renamed `plotting.plot_image` &rarr; `plotting.image`.
* Renamed `plotting.plot_profile` &rarr; `plotting.image_profiles`.
* Renamed `plotting.image_profiles` now as arguments 'x', and 'y' instead of 'xcoords' and 'ycoords'.
* Renamed `utils.get_centers` &rarr; `utils.centers_from_edges`.
* Renamed `utils.get_edges` &rarr; `utils.edges_from_centers`.
* Renamed `utils.rand_rows` &rarr; `utils.random_selection`.


## [0.0.6] - 2022-12-11

* Added call signatures for `image.make_slice_contour` and `image.make_slice_ellipsoid`; they are not yet implemented.
* Contour slices now take parameters `lmin` and `lmax`, which determine the thresholding window. This can be used to define either filled or shell slices.
* Internal changes to `image.project1d_contour` and `image.project2d_contour`.
* Fixed incorrect `plotting.corner` limits when diag=False.


## [0.0.5] - 2022-12-10

* Interactive plots `plotting.interactive_proj1d` and `plotting.interactive_proj1d` can now handle arrays of any dimension.
* Changed default behaviour of `plotting.interactive_proj2d_discrete` to use auto binning.
* Removed fix_vmax checkbox from `plotting.interactive_proj2d`.


## [0.0.4] - 2022-12-09

* Added `utils.get_edges`
* Diagonal plots in `plotting.corner` now fill in masked values.
* Fixed limits in `plotting.corner` diagonals.
* Renamed `utils.copy_into_new_dim` &rarr; `image.copy_into_new_dim`.
* Renamed `image.get_bin_centers` &rarr; `utils.get_centers`.
* Renamed `utils.save_stacked_array` &rarr; `data.save_stacked_array`.
* Renamed `utils.load_stacked_arrays` &rarr; `data.load_stacked_arrays`.
* Renamed `plotting.matrix_slice` &rarr; `plotting.slice_matrix`.


## [0.0.3] - 2022-12-06

* Added `bunch.decorrelate`: randomly permutes x-x', y-y', z-z' pairs.
* Fixed `plotting.plot_image`; thresholding no longer changes input array.
* Renamed `dist.py` &rarr; `bunch.py`.

## [0.0.2] - 2022-11-30
* Initial release.
