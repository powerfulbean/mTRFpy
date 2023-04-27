Regularization
==============

Ordinary least squares (OLS) regression minimizes the mean squared error (MSE) between predictions and observations. In the two-dimensional case that means finding the line that best fits a set of points. However, in practice, this often doesn't have the desirable outcome because the sampled data is not a perfect representation of the overall population. Thus, OLS optimizes the models fit on the sample to the detriment of its generalizability. To overcome this, one may use a regularization parameter :math:`\lambda` that penalizes large coefficients. In the below example, we use simulated data to demonstrate the effect of regularization. We apply regularized regression to four data points (filled circles) sampled from a larger set (unfilled circles). 

.. image:: images/reg.png
    :align: center
    :scale: 35 %

When :math:`\lambda=0`, this is equivalent to OLS regression (yellow line) and gives the best fit to the sample at the cost of deviating from the overall trend in the data (dashed black line). As the value of :math:`\lambda` is increased the line flattens because the slope is penalized. As lambda approaches infinity the regression produces a horizontal line, irrespective of the data. The optimal :math:`\lambda` value is the one that gives the best approximation of the population trend.
This value depends on multiple factors like the amount and quality of the data (more data requires less regularization) and the number of model parameters (larger models require more regularization) and has to be estimated from the data.

Optimization
------------

To optimize :math:`\lambda`, we must try out different values and choose the one that gives us the model that best predicts the actual data (i.e. that minimizes the MSE between predicted and observed data). Thus, we estimate the models accuracy for each candidate value of :math:`lambda` using cross-validation, pick the best one, and use it to fit a model on the whole data. This whole procedure can be done by using the `TRF.train` function and passing a list instead of a single value for the :py:const"`regularization` parameter. When testing multiple values for :math:`\lambda`, :py:meth:`TF.train` will return the correlation coefficient and MSE for each value. In the below example we are using those estimates to visualize how the model's accuracy changes as a function of :math:`\lamda` ::

    import numpy as np
    from matplotlib import pyplot as plt
    from mtrf.model import TRF, load_sample_data
    trf = TRF()  # use forward model
    stimulus, response, fs = load_sample_data(n_segments=10) # data will be downloaded
    tmin, tmax = 0, 0.4  # range of time lags
    regularization = np.logspace(-1, 6, 20)
    correlation, error = trf.train(
        stimulus, response, fs, tmin, tmax, regularization, k=-1
        )
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.semilogx(regularization, correlation, color='c')
    ax2.semilogx(regularization, error, color='m')
    ax1.set(xlabel='Regularization value', ylabel='Correlation coefficient')
    ax2.set(ylabel='Mean squared error')
    ax1.axvline(regularization[np.argmin(error)], linestyle='--', color='k')
    plt.show()

.. image:: images/opt.png

The dashed line marks the :math:`lambda` value where the MSE (purple line) was lowest. Usually, this will also be the point where the correlation is highest but there can be cases where MSE and correlation deviate.

Overfitting
-----------
Optimizing `\lambda` means that we are picking the value that generates the best predictions for a given data set. Because that set is not a perfect representation of the general population, the accuracy for new, unseen, data is expected to be lower than the estimated one. This so called overfitting may result in an overestimation of model accuracy. To circumvent this, we can withhold part of the data from the optimization process and use that data to test the final model. To avoid arbitrarily selecting this testing set, we can run two nested cross-validation loops where the data is split into three segments - training, validation and testing. For each split, training and validation set are used to optimize the model using cross-validation (as in the previous section) and the model's accuracy is estimated using the testing set. The data is rotated so that each subset is used for testing once and the final accuracy estimate is obtained by averaging across all splits. In the below example we are performing this procedure using the :py:meth:`TRF.test` method which takes the same arguments as :py:meth:`TRF.train` and returns the correlation, MSE and optimal :math:`\lambda` for each testing data segment::

    r, mse, best_reg = trf.test(
        stimulus, response, fs, tmin, tmax, regularization, k=-1
        )
    print(f'Correlation between the actual and predicted response is {r.mean().round(3)}')

    out:

    Correlation between the actual and predicted response is 0.024

The average correlation is an unbiased estimate of the model's accuracy in the sense that the data used for testing was never part of the model fitting procedure. Note however, that `TRF.test` will not give a single answer regarding the best value for `\lambda` because the optimal value might vary across the different segmentations of the data.


Regularization Methods
----------------------
All previous examples used the default ridge regularization which penalizes large model weights. Another method is Tikhonov regularization which penalizes the first derivative (i.e. the change in) model weights, providing a temporally smoothed result [#f1]_. The regularization method is determined by the :py:const:`method` parameter, when creating an instance of the :py:class:`TRF` class. Yet another method is banded ridge regression which uses ridge regression but estimates :math:`\lambda` separately for different feature bands. This can be useful in multivariate models which combine discrete and continuous features. When using banded ridge you must provide the fit function with an additional :py:const:`bands` parameter denoting the size of the feature bands for which :math:`\lambda` is optimized. In the example below, we are computing a multivariate TRF with a 16-band spectrogram and the acoustical onsets (i.e. the half-wave rectified derivative of the envelope). We want to use the same :math:`\lambda` for all bands of the spectrogram and a separate :math:`\lambda` for the onsets so the band sizes are 16 and 1, respectively. The optimal values for :math:`\lambda` can be found in the diagonal of the regularization matrix stored in the :py:attr:`TRF.regularization` parameter ::
    
    trf = TRF(method='banded')
    onsets = [np.diff(s.mean(axis=1), prepend=[0]) for s in stimulus]
    for i, _ in enumerate(onsets):  # half-wave rectification
        onsets[i][onsets[i]<0] = 0
    combined = [np.vstack([s.T, o]).T for s, o in zip(stimulus, onsets)]
    regularization = np.logspace(-1, 5, 5)
    trf.train(combined, response, fs, tmin, tmax, regularization, bands=[16,1])
    print(f'optimal values for \u03BB: \n {np.diagonal(trf.regularization)[:18]}')

    out:

    optimal values for λ:
     [0.e+00 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05
     1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e+05 1.e-01]

The first value is 0 and corresponds to the models bias term which is not regularized. The next 16 values are the optimal :math:`\lambda` for the spectrogram and the last value is the optimal :math:`\lambda` for the acoustic onsets. Note that banded ridge increases the number of parameters (by 1 for each band) and thus makes the model more susceptible to overfitting. Also, computation time increases exponentially with the number of bands because all combinations of :math:`\lambda` are tested.

.. [#f1] Crosse, M. J., Zuk, N. J., Di Liberto, G. M., Nidiffer, A. R., Molholm, S., & Lalor, E. C. (2021). Linear modeling of neurophysiological responses to speech and other continuous stimuli: methodological considerations for applied research. Frontiers in Neuroscience, 1350.



