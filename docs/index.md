# psdist

This repository is a collection of analysis and visualization methods for position-momentum space (phase space) distributions. Its primary use has been to slice and project high-dimensional data, both in point-cloud and image format.


## Installation

https://github.com/austin-hoover/psdist

https://pypi.org/project/psdist/


## Examples

Some accelerator physics publications that use methods from this repository:
* https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.23.124201
* https://arxiv.org/abs/2301.04178

An example notebook is here: https://github.com/austin-hoover/psdist/tree/main/examples. 

Below are some example figures.


### Interactive slicing

<video src="https://user-images.githubusercontent.com/58012334/242989106-0ad88e3a-7b10-48d4-9f36-ff8581281e80.mov" controls="controls" style="max-width: 730px;">
</video>


### Slice matrix

The following figure represents a four-dimensional slice of a measured five-dimensional phase space distribution, with lower-dimensional projections shown on the side panels.

![](figures/slice_matrix.png)


### Shell slices

Shell slices can be used to visualize the dependence of the distribution in a low-dimensional subspace on the radius in some other high-dimensional subspace. The figure below selects "shells" based on four-dimensional density contours in the $x$-$x'$-$y$-$y'$ plane, then plots the distribution along another dimension ($w$) within each shell.

<img src="figures/shell_slice.png" width="50%">


### Corner plot

The classic corner plot shows all one- and two-dimensional projections of the distribution.

![](figures/corner_log.png)