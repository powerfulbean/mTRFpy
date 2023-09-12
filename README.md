![Package](https://github.com/powerfulbean/mTRFpy/workflows/Python%20package/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/powefulbean/mTRFpy/graphs/commit-activity)
[![Documentation Status](https://readthedocs.org/projects/mtrfpy/badge/?version=latest)](https://mtrfpy.readthedocs.io/en/latest/?badge=latest)
![PyPI pyversions](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![PyPI license](https://img.shields.io/badge/license-MIT-brightgreen)
[![PyPI version](https://badge.fury.io/py/mtrf.svg)](https://badge.fury.io/py/mtrf)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05657/status.svg)](https://doi.org/10.21105/joss.05657)
# mTRFpy - multivariate linear modeling

This is an adaptation of the matlab mTRF-toolbox using only basic Python and Numpy.
It aims to implement the same methods as the original toolbox and advance them.
This documentation provides tutorial-like demonstrations of the core functionalities like model fitting, visualization and optimization as well as a comprehensive reference documentation.


# Installation

You can get the stable release from PyPI:
```sh
    pip install mtrf 
```
    
Or get the latest version from this repo:
```sh
    pip install git+https://github.com/powerfulbean/mTRFpy.git
```

While mTRFpy only depends on numpy, matplotlib is an optional dependency used to
visualize models. It can also be installed via pip:

```sh
    pip install matplotlib
```

We also provide an optional interface to MNE-Python so it might be useful to [install mne](https://mne.tools/stable/install/manual_install.html) as well.

# Getting started

For a little tutorial on the core features of mTRFpy, have a look at our [online documentation](https://mtrfpy.readthedocs.io)
# Found a bug?

1. Please use the issue search to check if the issue has already been reportet.
2. Try to reproduce problem using the latest` master` branch.
3. Create an issue with a minimal example that reproduces the problem.

# Missing a feature?

Feature requests are welcome. But take a moment to find out whether your idea
fits with the scope and aims of the project. It's up to *you* to make a strong
case to convince the project's developers of the merits of this feature. Please
provide as much detail and context as possible.

# Want to contribute to the project?

Great! Please take a moment to read the ![contribution guidelines](https://github.com/powerfulbean/mTRFpy/blob/master/CONTRIBUTING.md) before you do.

# Citing mTRFpy
Bialas et al., (2023). mTRFpy: A Python package for temporal response function analysis. Journal of Open Source Software, 8(89), 5657, https://doi.org/10.21105/joss.05657
```
@article{Bialas2023,
    doi = {10.21105/joss.05657},
    url = {https://doi.org/10.21105/joss.05657},
    year = {2023}, publisher = {The Open Journal},
    volume = {8},
    number = {89},
    pages = {5657},
    author = {Ole Bialas and Jin Dou and Edmund C. Lalor},
    title = {mTRFpy: A Python package for temporal response function analysis},
    journal = {Journal of Open Source Software} } 
```



