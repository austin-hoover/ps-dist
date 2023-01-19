# psdist


This repository is a collection of analysis and visualization methods for six-dimensional position-momentum space (phase space) distributions.

The state of a classical system of particles in $n$-dimensional space is given by the coordinates of each particle in $2n$-dimensional phase space: $x$, $p_x$, $y$, $p_y$, $z$, $p_z$. Alternatively, some problems are formulated in terms of a smooth distribution function $f(x, p_x, y, p_y, z, p_z)$ that gives the density at each point in phase space. 

The phase space coordinates of individual particles are often known in computer simulations. Individual particles can be measured in some cases (https://www.esa.int/Science_Exploration/Space_Science/Gaia); in other cases, all that can be measured is a $2n$-dimensional histogram (image) that approximates the distribution function (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.064804).


## Examples

Some examples in accelerator physics that use methods from this repository:
* https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.23.124201
* https://arxiv.org/abs/2301.04178


![](https://github.com/austin-hoover/psdist/examples/figures/view_yyp_slice_xxp_wslice.gif)


## Installation

https://pypi.org/project/psdist/
