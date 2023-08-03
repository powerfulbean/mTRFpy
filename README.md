![Package](https://github.com/powerfulbean/mTRFpy/workflows/Python%20package/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/powefulbean/mTRFpy/graphs/commit-activity)
[![Documentation Status](https://readthedocs.org/projects/mtrfpy/badge/?version=latest)](https://mtrfpy.readthedocs.io/en/latest/?badge=latest)
![PyPI pyversions](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![PyPI license](https://img.shields.io/badge/license-MIT-brightgreen)
[![PyPI version](https://badge.fury.io/py/mtrf.svg)](https://badge.fury.io/py/mtrf)

mTRFpy - multivariate linear modeling
=====================================
This is an adaptation of the matlab mTRF-toolbox using only basic Python and Numpy. It aims to implement the same methods as the original toolbox and advance them. This documentation provides tutorial-like demonstrations of the core functionalities like model fitting, visualization and optimization as well as a comprehensive reference documentation.

Installation
------------
You can get the stable release from PyPI::
    
    pip install mtrf 

Or get the latest version from this repo::

    pip install git+https://github.com/powerfulbean/mTRFpy.git 

While mTRFpy only depends on numpy, matplotlib is an optional dependency used to
visualize models. It can also be installed via pip::

    pip install matplotlib

We also provide an optional interface to MNE-Python so it might be useful to [install mne](https://mne.tools/stable/instal/manual_install.html) as well.

Getting Started
---------------
For a little tutorial on the core features of mTRFpy, have a look at our [online documentation](https://mtrfpy.readthedocs.io)




