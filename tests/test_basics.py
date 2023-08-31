from pathlib import Path
import numpy as np
from mtrf.model import lag_matrix, TRF, load_sample_data
from mtrf.matrices import _check_data

root = Path(__file__).parent.absolute()


def test_check_data():
    n = np.random.randint(2, 10)
    stimulus, response, _ = load_sample_data(n_segments=n)
    stimulus, response, n_trials = _check_data(
        stimulus=stimulus, response=response, min_len=n, crop=False
    )
    assert n_trials == n
    stimulus, response, _ = _check_data(
        stimulus=stimulus, response=response, min_len=n, crop=True
    )
    stimulus, _, _ = _check_data(
        stimulus=stimulus, response=None, min_len=n, crop=False
    )
    assert all([s.shape == stimulus[0].shape for s in stimulus])
    assert all([r.shape == response[0].shape for r in response])
    _, response, _ = _check_data(
        stimulus=None, response=response, min_len=n, crop=False
    )
    np.testing.assert_raises(ValueError, _check_data, stimulus, response, n + 1)
    np.testing.assert_raises(ValueError, _check_data, stimulus, response[:-1], n)


def test_lag_matrix():
    stimulus, response, fs = load_sample_data(n_segments=1)
    stimulus = stimulus[0]
    for _ in range(10):
        tmin = np.random.randint(-200, -100) / 1e3
        tmax = np.random.randint(150, 500) / 1e3
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        lag_mat = lag_matrix(stimulus, lags, True, True)
        assert lag_mat.shape[0] == stimulus.shape[0]
        assert lag_mat.shape[1] == len(lags) * stimulus.shape[1] + 1
        lag_mat = lag_matrix(stimulus, lags, True, False)
        assert lag_mat.shape[0] == stimulus.shape[0]
        assert lag_mat.shape[1] == len(lags) * stimulus.shape[1]
        lag_mat = lag_matrix(stimulus, lags, False, False)
        assert stimulus.shape[0] - lag_mat.shape[0] == len(lags) - 1
        assert lag_mat.shape[1] == len(lags) * stimulus.shape[1]
        lag_mat = lag_matrix(stimulus, lags, False, True)
        assert stimulus.shape[0] - lag_mat.shape[0] == len(lags) - 1
        assert lag_mat.shape[1] == len(lags) * stimulus.shape[1] + 1


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
