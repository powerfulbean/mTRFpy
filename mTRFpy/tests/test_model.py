from pathlib import Path
import numpy as np
from numpy.random import randint, uniform
from scipy.io import loadmat
from mTRFpy.Model import TRF, cross_validate
root = Path(__file__).parent.absolute()


def test_train_forward():
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = speech_response['resp'][0:100]
    stimulus = speech_response['stim'][0:100]
    for i in range(5):
        reps = np.random.randint(2, 10)
        stimuli = np.stack([stimulus for _ in range(reps)])
        responses = np.stack([response for _ in range(reps)])
        tmin = np.random.uniform(-0.1, 0.05)
        tmax = np.random.uniform(0.1, 0.4)
        direction = np.random.choice([1, -1])
        regularization = np.random.uniform(0, 10)
        trf1 = TRF(direction=direction)
        trf1.train(stimuli, responses, fs, tmin, tmax, regularization)
        trf2 = TRF(direction=direction)
        trf2.train(stimuli[0], responses[0], fs, tmin, tmax, regularization)
        if direction == 1:
            assert trf1.weights.shape[0] == stimuli.shape[-1]
            assert trf1.weights.shape[-1] == response.shape[-1]
        if direction == -1:
            assert trf1.weights.shape[0] == stimuli.shape[-1]
            assert trf1.weights.shape[-1] == response.shape[-1]
        np.testing.assert_almost_equal(trf1.weights, trf2.weights, 10)


def test_predict():
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = speech_response['resp']
    stimulus = speech_response['stim']
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    regularization = np.random.uniform(0, 10)
    trf = TRF()
    trf.train(stimulus, response, fs, tmin, tmax, regularization)
    reps = np.random.randint(2, 10)
    stimuli = np.stack([stimulus for _ in range(reps)])
    responses = np.stack([response for _ in range(reps)])
    predictions = trf.predict(stimuli)
    assert predictions.shape == responses.shape
    for p in range(predictions.shape[0]-1):
        np.testing.assert_equal(predictions[p], predictions[p+1])
    predictions, correlations, error = trf.predict(stimuli, responses)
    assert np.isscalar(correlations) and np.isscalar(error)
    predictions, correlations, error = trf.predict(
            stimuli, responses, average_trials=False)
    assert correlations.shape[0] == error.shape[0] == reps
    
    predictions, correlations, error = trf.predict(
            stimuli, responses, average_features=False)
    assert correlations.shape[-1] == trf.weights.shape[-1]

    features = \
        [randint(trf.weights.shape[0]) for _ in range(randint(2, 10))]
    lags = \
        [randint(len(trf.times)) for _ in range(randint(2, 10))]
    predictions, correlations, error = trf.predict(
        stimuli, responses, lags, features)
    assert predictions.shape == responses.shape


def test_crossval():
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = np.stack([speech_response['resp'][0:100] for _ in range(20)])
    stimulus = np.stack([speech_response['stim'][0:100] for _ in range(20)])
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    reg = np.random.uniform(0, 10)
    trf = TRF(direction=direction)
    splits = np.random.randint(2, 10)
    test_size = np.random.uniform(0.05, 0.3)
    models, correlations, errors = cross_validate(
        trf, stimulus, response, fs, tmin, tmax, reg, splits, test_size)
