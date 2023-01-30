from pathlib import Path
import numpy as np
from mtrf.model import lag_matrix, TRF

root = Path(__file__).parent.absolute()


def test_lag_matrix():
    speech_response = np.load(
        root / "data" / "speech_data.npy", allow_pickle=True
    ).item()
    fs = speech_response["samplerate"][0][0]
    stimuli = speech_response["stimulus"]
    for _ in range(10):
        tmin = np.random.randint(-200, -100) / 1e3
        tmax = np.random.randint(150, 500) / 1e3
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        lag_mat = lag_matrix(stimuli, lags, True, True)
        assert lag_mat.shape[0] == stimuli.shape[0]
        assert lag_mat.shape[1] == len(lags) * stimuli.shape[1] + 1
        lag_mat = lag_matrix(stimuli, lags, True, False)
        assert lag_mat.shape[0] == stimuli.shape[0]
        assert lag_mat.shape[1] == len(lags) * stimuli.shape[1]
        lag_mat = lag_matrix(stimuli, lags, False, False)
        assert stimuli.shape[0] - lag_mat.shape[0] == len(lags) - 1
        assert lag_mat.shape[1] == len(lags) * stimuli.shape[1]
        lag_mat = lag_matrix(stimuli, lags, False, True)
        assert stimuli.shape[0] - lag_mat.shape[0] == len(lags) - 1
        assert lag_mat.shape[1] == len(lags) * stimuli.shape[1] + 1


def test_arithmatic():
    trf1 = TRF()
    trf2 = TRF()
    trf1.weights = np.ones((10, 10))
    trf2.weights = np.ones((10, 10))
    trf1.bias = np.ones(10)
    trf2.bias = np.ones(10)
    trf3 = trf1 + trf2
    np.testing.assert_equal(trf3.bias, trf1.bias + trf2.bias)
    np.testing.assert_equal(trf3.weights, trf1.weights + trf2.weights)
    trf4 = sum([trf1, trf2, trf3])
    np.testing.assert_equal(trf4.bias, trf1.bias + trf2.bias + trf3.bias)
    np.testing.assert_equal(trf4.weights, trf1.weights + trf2.weights + trf3.weights)
    trf4 /= 4
    np.testing.assert_equal(trf4.weights, trf1.weights)
