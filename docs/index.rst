Welcome to mTRFpy
=================

The mTRFpy package offers convenient tools for temporal response functions (TRFs) which are multivariate regression models for continuous data, commonly used to model neural responses to naturalistic speech. mTRFpy is an adaptation and advancement of the `MATLAB mTRF-toolbox <https://github.com/mickcrosse/mTRF-Toolbox.git>`_. This documentation provides guidance on the basic usage as well as questions of optimization and statistical inference along with a comprehensive reference documentation.


Installation
------------
You can get the stable release from PyPI::

    pip install mtrf

Or get the latest version from Github::

    pip install git+https://github.com/powerfulbean/mTRFpy.git

While mTRFpy only depends on numpy, matplotlib is an optional dependency used to
visualize models. It can also be installed via pip::

    pip install matplotlib

.. toctree::
  :caption: Contents
  :maxdepth: 2

  basics
  optimization
  inference
  api


Frequently Asked Questions
--------------------------
* **will mTRFpy produce the same results as the matlab mTRF-toolbox?**

Yes, the basic operations like :meth:`TRF.train()` and :meth:`TRF.predict()` will produce identical results (within the boundaries of computer numerical accuracy).


* **I think I found a bug!**

Please see the `bug reports <https://github.com/powerfulbean/mTRFpy/blob/master/CONTRIBUTING.md#bugs>`_ section in the contribution guidelines.


* **How can I contribute to the project?**

Please see the `pull request <https://github.com/powerfulbean/mTRFpy/blob/master/CONTRIBUTING.md#pull-requests>`_ section in the contribution guidelines if you want to contribute code or useful examples for the documentation.
