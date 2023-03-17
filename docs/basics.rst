Tutorial
========

The multivariate temporal response function is regression model which estimates a linear mapping between a stimulus and a neural response [#f1]_. Because the neural response is delayed with respect to the stimulus, the regression is computed across a series of time lags. The range of lags is determined by the parameters :attr:`tmin` and :attr:`tmax` and the step-size by the sampling rate :attr:`fs` of stimulus and response. It can be used as a forward model, predicting neural activation patterns in response to a stimulus, or as a backward model, reconstructing a stimulus from neural activity. This is determined by the :attr:`direction` parameter.

Training models
---------------
We can fit a simple forward model using the mTRFpy sample dataset, consisting of neural responses to about 2 minutes of naturalistic speech, recorded with a 128 channel EEG system, as well as the 16-band spectrogram of that speech.::
    
    from mtrf.model import TRF, load_sample_data
    
    stimulus, response, fs = load_sample_data() # data will be downloaded
    trf = TRF(direction=1)
    tmin, tmax = 0, 0.4  # range of time lags
    trf.train(stimulus, response, fs, tmin, tmax, regularization=1)

The last parameter is the regularization parameter, which is addressed in a later section. Now, we can use the fitted TRF to predict the neural response from the stimulus and compare the prediction to the actual response to assess the model's accuracy.::

    prediction, correlation, error = trf.predict(stimulus, response)

We can use a backward model by simply changing the :attr"`direction` parameter to -1.
::

    trf = TRF(direction=-1)
    tmin, tmax = 0, 0.4  # range of time lags
    trf.train(stimulus, response, fs, tmin, tmax, regularization=1)
    prediction, correlation, error = trf.predict(stimulus, response)

The correlation between the predicted and actual output is 0.76, which is impossibly high given the noisy nature of EEG data. This is clear overfitting, resulting from training and testing on the same data. However, cross-validation can give an unbiased estimate of the model's accuracy.

Cross-validation
----------------
For cross-validation, data are split into k subsets. All but one of these subsets are used to train the TRF while the last subset is used to test the TRF's prediction. This is repeated k times so that each subset is used for testing once. The average correlation across all splits is an unbiased estimate of the model's accuracy.::

    import numpy as np
    from mtrf.crossval import cross_validate
    trf = TRF(direction=-1)
    stimulus, response, fs = load_sample_data() # data will be downloaded
    # split stimulus and response into 10 segments
    stimulus = np.array_split(stimulus, 10)
    response = np.array_split(response, 10)
    correlation, error = cross_validate(  # this will take a few moments
        trf, stimulus, response, fs, tmin, tmax, regularization=1, k=5
        )

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
