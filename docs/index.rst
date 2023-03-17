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

We also provide an optional interface to MNE-Python so it might be useful to `install mne <https://mne.tools/stable/install/manual_install.html>`_ as well.

.. toctree::
  :caption: Contents
  :maxdepth: 2

  basics
  api


Frequently Asked Questions
--------------------------
* **will mTRFpy produce the same results as the matlab mTRF-toolbox?**

Yes, the basic operations like ``TRF.train()`` and ``TRF.predict()`` will produce identical results (within the boundaries of computer numerical accuracy).


* **I think I found a bug!**

Please see the `bug reports <https://github.com/user/powerfulbean/mTRFpy/CONTRIBUTING.md#bugs>`_ section in the contribution guidelines.


* **How can I contribute to the project?**

Please see the `pull request <https://github.com/user/powerfulbean/mTRFpy/CONTRIBUTING.md#pull-requests>`_ section in the contribution guidelines if you want to contribute code or useful examples for the documentation.
