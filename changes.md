<!-- ## [0.0.0] - YYYY-MM-DD
   
### Added
* Item 1
* Item 2
 
### Changed
* Item 1

### Fixed
* Item 1 -->


# Change Log 

## [0.0.6] - Unreleased

### Changed
* Contour slices now take parameters `lmin` and `lmax`, which determine the thresholding window. This is very general, and can be used to define either filled or shell slices. (This is currently only done in `image.project1d_contour` and `image.project2d_contour`. I have written the signature for a function `image.make_slice_contour`, but the function is not implemented yet. I haven't figured out how to construct the correct mask in the general case.)

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
