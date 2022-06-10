from pathlib import Path
import numpy as np
from scipy.io import loadmat
from mTRFpy.Model import TRF, cross_validate
root = Path(__file__).parent.absolute()


def test_train():
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = speech_response['resp']
    stimulus = speech_response['stim']
    for i in range(5):
        reps = np.random.randint(2, 10)
        stimuli = np.stack([stimulus for _ in range(reps)])
        responses = np.stack([response for _ in range(reps)])
        tmin = np.random.uniform(-0.1, 0.05)
        tmax = np.random.uniform(0.1, 0.4)
        regularization = np.random.uniform(0, 10)
        trf1 = TRF()
        trf1.train(stimuli, responses, fs, tmin, tmax, regularization)
        trf2 = TRF()
        trf2.train(stimuli[0], responses[0], fs, tmin, tmax, regularization)
        np.testing.assert_almost_equal(trf1.weights, trf2.weights, 11)


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
    for i in range(5):
        reps = np.random.randint(2, 10)
        stimuli = np.stack([stimulus for _ in range(reps)])
        responses = np.stack([response for _ in range(reps)])
        predictions = trf.predict(stimuli)
        assert predictions.shape == responses.shape
        for p in range(predictions.shape[0]-1):
            np.testing.assert_equal(predictions[p], predictions[p+1])




"""
def test_cross_validation():
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = speech_response['resp']
    stimulus = speech_response['stim']
    reps = 5
    stimuli = np.stack([stimulus for _ in range(reps)])
    response = np.stack([response for _ in range(reps)])
    model = TRF()
    tmin, tmax = -0.1, 0.2
    regularization = 0.1
    splits = 5
    test_size = 0.1
    cross_validate(model, stimuli, response, fs,
                   tmin, tmax, regularization, splits, test_size)
"""
