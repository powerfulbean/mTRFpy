===========
Basic usage
===========

This section describes the basic methods to compute a temporal response function on continuous data, estimate its predictive accuracy and visualize the results.

Background
==========
The TRF is a multivariate regression between a continuous stimulus and its neural response. The TRF can be applied as a forward model, using stimulus features to predict neural responses or as a backward model, using neural responses to reconstruct the stimulus. Because neural responses are delayed with respect to the stimulus, the regression must be computed across multiple time lags. The TRF predicts the estimand (i.e. neural response in a forward or stimulus in a backward model) as a weighted combination of predictor features. The model weights are chosen to minimize the mean squared error (MSE) between estimand and it's prediction. This is implemented in the following matrix multiplication:

.. math::

    w = (X^\intercal X+\lambdaI)^{-1}X^\intercal y

Where X is a matrix of time-lagged input features and y is a vector of output features.
:math:`(X^\intercal X)^{-1}` is the inverse autocovariance matrix of the predictor which accounts for the fact that both natural speech and brain signals are correlated with themselves over time. :math:`\lambdaI` is a diagonal matrix with a regularization parameter that can be optimized to improve stability and avoid overfitting.  :math:`X^\intercal y` is the covariance matrix of predictor and estimand [#f1]_. 

Because the TRF minimized the difference between the estimand and it's prediction within the training data set we must test wheter the estimated linear mapping generalizes. To do this, we use k-fold cross-validation, where the data is split into k subsets, a TRF is trained on k-1 of them and validated on the kth one. The data are rotated so that each segment is used for validation and the models accuracy can be obtained as the correlation or mean squared error averaged across all k validation sets.


Sample dataset
==============
We provide a small sample dataset for testing purposes. The data contains about two minutes of one individual's brain responses, recorded with a 128 biosemi EEG system while they were listening to an audiobook. It can be obtained by calling a function which will download the data if they are not present. ::
    
    from mtrf.model import load_sample_data

    stimulus, response, fs = load_sample_data()

In the above example, ``stimulus`` contains a 16-band spectrogram of the speech signal and ``response`` contains the 128-channel EEG recording. Both are resampled to 128 Hz and the sampling rate is stored in the variable ``fs``.


Forward model
=============

To fit a forward model, create an instance of the TRF class, define the time interval (``tmin`` and ``tmax``) and the ``regularization`` value (details about regularization can be found in the next section) and use the ``train()`` method. ::

    
    from mtrf.model import TRF, load_sample_data

    stimulus, response, fs = load_sample_data() # data will be downloaded
    fwd_trf = TRF(direction=1)
    tmin, tmax = 0, 0.4  # range of time lags
    regularization = 1
    fwd_trf.train(stimulus, response, fs, tmin, tmax, regularization)

Now, we can use the fitted TRF to predict the neural response from the stimulus and compare the prediction to the actual response to assess the model's accuracy.::

    prediction, r_fwd, mse_fwd = fwd_trf.predict(stimulus, response)
    print(f"correlation between actual and predicted response: {r_fwd.round(3)}")

    out:
    correlation between actual and predicted response: 0.124

However, because we trained and tested on the exact same data, the correlation coefficient is inflated due to overfitting. To avoid this we can use k-fold cross-validation which will give an unbiased accuracy estimate for a model with a given set of parameters. The ``cross_validate()`` function takes a ``TRF`` object as input and uses the same parameters as the ``train()`` method. Additionally, you can specify the number of folds ``k``. The default value is -1 which corresponds to leave-one-out cross-validation where the number of folds is equal to the number of trials. To use cross-validation we must split the data into multiple trials which can be done using numpy's ``array_split()`` function. ::

    import numpy as np
    from mtrf.stats import cross_validate
    stimulus = np.array_split(stimulus, 10)
    response = np.array_split(response, 10)
    r, mse = cross_validate(fwd_trf, stimulus, response, fs, tmin, tmax, regularization)
    print(f"correlation between actual and predicted response: {r_fwd.round(3)}")

    out:
    correlation between actual and predicted response: 0.018

Turns out the first estimate of the model's accuracy was about an order of magnitude too large!

Backward model
==============
Fitting a backward model works in the same way, just set the ``direction`` parameter to -1. To save time, we won't compute the TRF for each spectral band but rather average across bands to obtain the acoustic envelope. Because a single stimulus feature is reconstructed from 128 neural recordings, the backward model is more powerful but also more susceptible to overfitting. ::
    
    envelope = [s.mean(axis=1) for s in stimulus]
    bwd_trf = TRF(direction=-1)
    bwd_trf.train(envelope, response, fs, tmin, tmax, regularization)
    prediction, r_bwd, mse_bwd = bwd_trf.predict(envelope, response)
    print(f"correlation between actual and predicted response: {r_bwd.round(3)}")

    out:
    correlation between actual and predicted response: 0.1



Visualization
-------------


The TRF class has a plot method to quickly visualize the models' weights. Because the weight matrix is three-dimensional (inputs-by-lags-by-outputs) visualization requires selecting from or averaging across one of the dimensions.

.. plot::
    :include-source:
    
    import numpy as np
    from matplotlib import pyplot as plt
    from mtrf.model import TRF, load_sample_data
    tmin, tmax = 0, 0.4  # range of time lags
    trf = TRF()  # use forward model
    stimulus, response, fs = load_sample_data() # data will be downloaded
    trf.train(stimulus, response, fs, tmin, tmax, 1)
    fig, ax = plt.subplots(2)
    trf.plot(feature='avg', axes=ax[0], show=False)
    trf.plot(channel=60, axes=ax[1], kind='image', show=False)
    plt.tight_layout()
    plt.plot()

The top panel shows the TRF for each EEG-channel with the weights averaged across all features (i.e. spectral bands) and the second panels shows the TRF for each feature at a specific channel.

The TRF can also be easily converted to MNE-Pythons evoked class (requires that mne is installed) to access more visualization methods. Per default, this method creates one evoked response for each feature in the TRF

.. plot::
    :include-source:

    import numpy as np
    from matplotlib import pyplot as plt
    import mne 
    from mtrf.model import TRF, load_sample_data
    
    tmin, tmax = 0, 0.4  # range of time lags
    trf = TRF()  # use forward model
    stimulus, response, fs = load_sample_data() # data will be downloaded
    trf.train(stimulus, response, fs, tmin, tmax, 1)
    
    # use standard montage for the EEG system used for recording the response
    montage = mne.channels.make_standard_montage('biosemi128')
    evokeds = trf.to_mne_evoked(montage)
    evokeds[0].plot_joint([0.175, 0.26, 0.32], topomap_args={"scalings": 1}, ts_args={"units": "a.u.", "scalings": dict(eeg=1)})
    

The plot shows each channel's TRF for one spectral band as well as the distribution of TRF weights across the scalp. This is conceptually similar to an average evoked response potential or ERP.


.. [#f1] Crosse, M. J., Di Liberto, G. M., Bednar, A., & Lalor, E. C. (2016). The multivariate temporal response function (mTRF) toolbox: a MATLAB toolbox for relating neural signals to continuous stimuli. Frontiers in human neuroscience, 10, 604.
