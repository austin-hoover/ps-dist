<!-- ## [0.0.0] - YYYY-MM-DD
   
### Added
* Item 1
* Item 2
 
### Changed
* Item 1

### Fixed
* Item 1 -->


# Change Log 

## [0.0.9] - Unreleased

### Added
* Updated examples

### Fixed
* Fixed typo in `plotting.corner`. There was a lingering variable named 'image', which produced an error after changing the function `plot_image` to `image`. 
* `utils.random_selection` can now take lists as input.


## [0.0.8] - 2022-12-15

### Fixed
* `plotting.image_profiles` was flipping axis limits sometimes... this patch is a temporary fix.


## [0.0.7] - 2022-12-15

### Added
* `bunch.downsample` — removes a random selection of points.
* `image.enclosing_ellipsoid` — scales covariance ellipsoid until it contains some fraction of points.
* `image.enclosing_sphere` — scales sphere until it contains some fraction of points.
* `plotting.corner` can now act on existing axes (allowing more than one data set to be plotted).
* `plotting.ellipse` — convenience function to plot an ellipse.
* `plotting.rms_ellipse` — plots rms ellipse at various levels given covariance matr
* `plotting.image_rms_ellipse` — computes rms ellipse from image and plots it.
* 'divide_by_max' parameter in `plotting.image`
* Started an examples notebook.

### Changed
* Removed contour parameter from `plotting.plot_image`.
* `plotting.plot_image` &rarr; `plotting.image`.
* `plotting.plot_profile` &rarr; `plotting.image_profiles`.
* `plotting.image_profiles` now as arguments 'x', and 'y' instead of 'xcoords' and 'ycoords'.
* `utils.get_centers` &rarr; `utils.centers_from_edges`.
* `utils.get_edges` &rarr; `utils.edges_from_centers`.
* `utils.rand_rows` &rarr; `utils.random_selection`.

### Fixed
* `image.cov is now much faster for high-dimensional images.
* Threshold slider in `plotting.interactive_proj2d` now applies a fractional threshold instead of an absolute threshold.
* Fixed naming conflict in `plotting.interactive_proj2d_discrete`.
* Allow k=None in `utils.random_selection`


## [0.0.6] - 2022-12-11

### Added
* Signatures for `image.make_slice_contour` and `image.make_slice_ellipsoid`; they are not yet implemented.

### Changed
* Contour slices now take parameters `lmin` and `lmax`, which determine the thresholding window. This can be used to define either filled or shell slices.
* Internal changes to `image.project1d_contour` and `image.project2d_contour`.

### Fixed
* Corner plot limits were incorrect when diag=False.



## [0.0.5] - 2022-12-10

### Changed
* Removed fix_vmax checkbox from `plotting.interactive_proj2d`.
* Default is to use auto binning for `plotting.interactive_proj2d_discrete`.

### Fixed
* Interactive plots can now handle arrays of any dimension.


## [0.0.4] - 2022-12-09

### Added
* In `plotting.corner`, the 1D diagonal plots will fill in masked values.

### Changed
* `utils.copy_into_new_dim` &rarr; `image.copy_into_new_dim`.
* `image.get_bin_centers` &rarr; `utils.get_centers`.
* Added `utils.get_edges`.
* `utils.save_stacked_array` &rarr; `data.save_stacked_array`.
* `utils.load_stacked_arrays` &rarr; `data.load_stacked_arrays`.
* `plotting.matrix_slice` &rarr; `plotting.slice_matrix`.

### Fixed
* Limits on corner plot diagonals were not always correct.


## [0.0.3] - 2022-12-06

### Added
* `bunch.decorrelate`: randomly permutes x-x', y-y', z-z' pairs.

### Changed
* `dist.py` &rarr; `bunch.py`.

### Fixed
* `plotting.plot_image` thresholding no longer changes input array.


## [0.0.2] - 2022-11-30
* Initial release.
