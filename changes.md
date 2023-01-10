# Change Log


# [0.0.12] — Unreleased
* Added 'linefilled' and 'stepfilled' plot kinds in convenience function `plotting.lineplot`.


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
