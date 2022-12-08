<!-- ## [0.0.0] - YYYY-MM-DD
   
### Added
* Item 1
* Item 2
 
### Changed
* Item 1

### Fixed
* Item 1 -->


# Change Log 


## [0.0.4] - YYYY-MM-DD

### Changed
* Moved `copy_into_new_dim` function to `image` module.
* Renamed `get_bin_centers` function to `get_centers`; added `get_edges` function, moved both to `utils` module.
* Moved `stack_ragged`, `save_stacked_array`, and `load_stacked_arrays` to `data` module.


## [0.0.3] - 2022-12-06

### Added
* `bunch.decorrelate`: randomly permutes x-x', y-y', z-z' pairs.

### Changed
* `dist` module renamed to `bunch`

### Fixed
* `plotting.plot_image` thresholding no longer changes input array.


## [0.0.2] - 2022-11-30
* Initial release
