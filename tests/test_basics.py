from pathlib import Path
import numpy as np
import dask.array as da
import array_api_strict
from mtrf.matrices import _check_data
from mtrf.model import lag_matrix, TRF, load_sample_data


root = Path(__file__).parent.absolute()


def test_sample_data_segmentation():
    for _ in range(5):
        n = np.random.randint(1, 10)
        stimulus, response, _ = load_sample_data(n_segments=n)
        assert all([isinstance(data, list) for data in [stimulus, response]])
        assert len(stimulus) == len(response)
        assert all([len(s) == len(r) for s, r in zip(stimulus, response)])


def test_sample_data_normalization():
    stimulus, response, _ = load_sample_data(normalize=True)
    np.testing.assert_almost_equal(stimulus[0].mean(), 0, decimal=12)
    np.testing.assert_almost_equal(response[0].mean(), 0, decimal=12)
    np.testing.assert_almost_equal(stimulus[0].std(), 1, decimal=12)
    np.testing.assert_almost_equal(response[0].std(), 1, decimal=12)
    stimulus, response, _ = load_sample_data(normalize=False)
    np.testing.assert_almost_equal(stimulus[0].mean(), 0.11, decimal=1)
    np.testing.assert_almost_equal(response[0].mean(), -0.01, decimal=1)
    np.testing.assert_almost_equal(stimulus[0].std(), 0.15, decimal=1)
    np.testing.assert_almost_equal(response[0].std(), 8.69, decimal=1)


def test_array_namespace_is_returned():
    stimulus, _, _ = load_sample_data(n_segments=5)
    stimulus, xp = _check_data(stimulus)
    assert xp.__name__.split(".")[-1] == "numpy", "Numpy array wasn't detected!"
    stimulus = [da.from_array(s) for s in stimulus]
    stimulus, xp = _check_data(stimulus)
    assert xp.__name__.split(".")[1] == "dask", "Dask array wasn't detected!"
    stimulus = [array_api_strict.asarray(s) for s in stimulus]
    stimulus, xp = _check_data(stimulus)
    assert xp.__name__ == "array_api_strict", "Array-API array wasn't detected!"


def test_lag_matrix_size():
    stimulus, _, fs = load_sample_data(n_segments=1)
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


def test_trf_arithmetic():
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
