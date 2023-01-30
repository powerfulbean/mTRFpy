![Package](https://github.com/powerfulbean/mTRFpy/workflows/Python%20package/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/powefulbean/mTRFpy/graphs/commit-activity)
![PyPI pyversions](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![PyPI license](https://img.shields.io/badge/license-MIT-brightgreen)

# mTRFpy
This is an adaptation of the [matlab mTRF-toolbox](https://github.com/mickcrosse/mTRF-Toolbox
) implemented in basic Python and Numpy. It aims to provide the same functionalities as the original toolbox and eventually advance them. The package is written and maintained by Jin Dou and Ole Bialas at the University of Rochester.

# Installation
You can get the stable release from PyPI:\
``` 
pip install mtrf 
```

Or get the latest version from this repo:\
```
pip install git+https://github.com/powerfulbean/mTRFpy.git 
```

# Tutorial
Here, we provide an overview of mTRFpy's core functions

## The TRF class
The TRF class is the core of the toolbox, we import it along with some sample data (the data
will be downloaded when you call the loading function for the first time.

```python 
from mtrf.model import TRF, load_sample_data
# stimulus is a 16-band spectrogram, response a 128-channel EEG
stimulus, response, samplerate = load_sample_data()
trf = TRF(direction=1)  # create a forward TRF
```

The TRF is applied to the data using the train method which requires specification of the range of time lags and the regularization parameter, often called lambda. To test the models accuracy, we can use the trained TRF to predict the EEG from the stimulus and compute the correlation between the prediction and actual data.

```python
trf.train(stimulus, response, samplerate, tmin=0, tmax=0.3, regularization=1000)
# add the argument `average=False` to get one correlation coefficient per channel
prediction, correlation, error = trf.predict(stimulus, response)
print(f"Pearson's correlation between actual brain response and prediction: {correlation.round(3)}")
```

The TRF class also has a plotting method to visualize the weights across time. Using the trained TRF we could, for example, plot the weights for each spectral band at one channel or plot the weights for each channel, averaged across all spectral bands

```python
from matplotlib import pyplot as plt
fig, ax = plt.subplots(2)
trf.plot(channel=60, axes=ax[0], show=False, kind='line')
ax[0].set_title('16-band spectrogram TRF at channel 60')
trf.plot(feature='avg', axes=ax[1], show=False, kind='image')
ax[1].set_title('Average TRF at every channel')
plt.tight_layout()
plt.show()
```

## Prevent overfitting
TRFs can also be used as a backward model to the stimulus envelope (i.e. the average spectrogram) from the recorded EEG.

```python
trf = TRF(direction=-1) # create a backward TRF
envelope = stimulus.mean(axis=-1, keepdims=True)
trf.train(envelope, response, samplerate, tmin=0, tmax=0.3, regularization=1000)
prediction, correlation, error = trf.predict(envelope, response)
print(f"Pearson's correlation between actual envelope and prediction: {correlation.round(3)}")
```

The correlation between the predicted and actual envelope is 0.56, which is far too high. This is the result of overfitting because we are using a model with lots of free parameters (one per channel) and a single estimand (the envelope). To prevent overfitting we need to train the TRF on one (part of the) dataset and test it on another. This can be done systematically using the `cross_validate` function. To use it, we must reshape stimulus and response into a 3-D array of shape trials x samples x features.

```python
import numpy as np
from mtrf.crossval import cross_validate
# split stimulus and response into 10 trials
envelope, response = np.array_split(envelope, 10), np.array_split(response, 10)
correlation, error = cross_validate(TRF(direction=-1), envelope, response, samplerate, tmin=0, tmax=0.3, regularization=1000)
print(f"Pearson's correlation between actual envelope and prediction: {correlation.round(3)}")
```

The correlation estimated via cross-validation is a more accurate description of the decoders accuracy.

## Fitting hyperparameters
So far, we used a regularization value of 1000 in all examples which worked reasonably well, judging from the correlation values and visual inspection of TRFs. However, a more principled way is to find the regularization value yielding the most accurate predictions. This can be done using the `fit` method. This method takes a list of regularization values, creates a TRF-model for each one and tests its accuracy with cross validation. Then, the value yielding the highest correlation is selected to train the final model.

```python
trf = TRF(direction=1)  # create a forward TRF
regularization=np.logspace(-1, 6, 10)  # try 10 values between 0.1 and 1,000,000
stimulus = np.array_split(stimulus, 10)  # split stimulus as well
correlation, error = trf.fit(stimulus, response, samplerate, tmin=0, tmax=0.3, regularization=regularization)
```

The TRF class also implements banded ridge regression. This allows us to split our features into bands and fitting the regularization parameter to each band. When using this method, you need to define the bands as an argument of the `fit` method. For example, we could fit the regularization to the first and second half of the spectrogram separately (this is just for demonstration purposes, you would not actually do this). Note that the computational cost increases exponentially with the number of bands because the total number of iterations is defined by $n_{regularization}^{n_{bands}}$

```python
trf = TRF(direction=1, method='banded')  # create a forward TRF
bands = [8, 8]  # first and second half of the spectrogram
regularization=np.logspace(-1, 6, 5)  # only 5 values to reduce computation time
correlation, error = trf.fit(stimulus, response, samplerate, tmin=0, tmax=0.3, regularization=regularization, bands=bands)
```

Note that, fitting the regularization on the data that the model is being tested on also constitutes a (less severe) form of overfitting. To avoid this you should test the final model on data that was withheld from fitting.

# Found a bug or missing a feature?
If you want to report a bug or request the implementation of a feature, please take a moment to review the [guidelines for contributing](CONTRIBUTING.md).

* [Bug reports](CONTRIBUTING.md#bugs)
* [Feature requests](CONTRIBUTING.md#features)
* [Pull requests](CONTRIBUTING.md#pull-requests)

# License
The project is licensed under the BSD 3-Clause License.

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)





