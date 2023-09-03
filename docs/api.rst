
Reference documentation
=======================

.. note:: This reference documentation is auto-generated from the doc strings in the module. For a tutorial-like overview of the functionality of mtrf, please see the previous sections.


TRF model
^^^^^^^^^
.. autoclass:: mtrf.TRF
   :members:
   :member-order: bysource
.. autofunction:: mtrf.load_sample_data

Cross-validation
^^^^^^^^^^^^^^^^
.. autofunction:: mtrf.stats.crossval
.. autofunction:: mtrf.stats.nested_crossval
.. autofunction:: mtrf.stats.neg_mse
.. autofunction:: mtrf.stats.pearsonr

Matrix operations
^^^^^^^^^^^^^^^^^
.. autofunction:: mtrf.matrices.truncate
.. autofunction:: mtrf.matrices.covariance_matrices
.. autofunction:: mtrf.matrices.regularization_matrix
.. autofunction:: mtrf.matrices.banded_regularization
