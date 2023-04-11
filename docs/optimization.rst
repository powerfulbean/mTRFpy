Optimization and statistical inference
======================================

Regularization
--------------
One important parameter is the regularization value which penalizes the slope of the regression. Thus, a large regularization value prevents the model from 'overreacting' to large outlier values. The ``TRF.fit()`` method takes a list of regularization values, makes a model for each value, tests their accuracy using cross-validation and selects the regularization value that yield the best model.

.. plot::
    :include-source:
    
    import numpy as np
    from matplotlib import pyplot as plt
    from mtrf.model import TRF, load_sample_data
    trf = TRF()  # use forward model
    stimulus, response, fs = load_sample_data() # data will be downloaded
    stimulus = np.array_split(stimulus, 10)
    response = np.array_split(response, 10)
    tmin, tmax = 0, 0.4  # range of time lags
    regularization = np.logspace(-1, 6, 20)
    correlation, error = trf.fit(
        stimulus, response, fs, tmin, tmax, regularization, k=-1
        )
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.semilogx(regularization, correlation, color='c')
    ax2.semilogx(regularization, error, color='m')
    ax1.set(xlabel='Regularization value', ylabel='Correlation')
    ax2.set(ylabel='Mean squared error')
    ax1.axvline(regularization[np.argmin(error)], linestyle='--', color='k')
    plt.show()

The dashed line marks the regularization coefficient which yields the best TRF (i.e. the one that minimizes the mean squared error between predicted and actual response).
