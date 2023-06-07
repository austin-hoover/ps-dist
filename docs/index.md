The `psdist` repository is a collection of analysis and visualization methods for position-momentum space (phase space) distributions. Its primary use has been to slice and project high-dimensional data, both in point-cloud and image format.

A few example notebooks are here: https://github.com/austin-hoover/psdist/tree/main/examples.

Below are a some examples from my own research.


### Publications

* https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.23.124201
* https://arxiv.org/abs/2301.04178


### Interactive slicing

One-dimensional projections of a five-dimensional image:

<video src="https://user-images.githubusercontent.com/58012334/242989106-0ad88e3a-7b10-48d4-9f36-ff8581281e80.mov" controls="controls" style="max-width: 600px;">
</video>

Two-dimensional projections of a five-dimensional image:

<video src="https://user-images.githubusercontent.com/58012334/242990288-94b777a6-6f69-44f9-a11f-81ccda179594.mov" controls="controls" style="max-width: 600px;">
</video>

Similar functions are available for point cloud data. Passing more than one image to the function enables side-by-side comparisions of multiple high-dimensional distributions with interactive slicing.


### Slice matrix

The following figure represents a four-dimensional slice of a measured five-dimensional phase space distribution, with lower-dimensional projections shown on the side panels.

![](figures/slice_matrix.png)


### Shell slices

Shell slices can be used to visualize the dependence of a distribution in a low-dimensional subspace on the distance from the origin in a different high-dimensional subspace. The figure below selects "shells" based on four-dimensional density contours in the transverse phase space ($x$-$x'$-$y$-$y'$), then plots the distribution along another dimension ($w$) within each shell. The $w$ distribution transitions from unimodal at large transvere amplitude to bimodal in the core.

<img src="figures/shell_slice.png" width="50%">


### Corner plot

The classic corner plot shows all one- and two-dimensional projections of the distribution. The version in `psdist` can be used to plot both image and point-cloud data. It has a few other options like right-hand spines on the diagonal subplots to allow logarithmic scaling, which is important in some applications. The following plot is of a simulated ion beam with 8 million macroparticles transported through the SNS linac using the PyORBIT code. The colormaps have logarithmic normalization.

![](figures/corner_log.png)
