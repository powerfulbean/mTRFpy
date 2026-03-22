import math
import warnings
import numpy as np
from mtrf import stats
from mtrf.model import TRF, load_sample_data
from mtrf.stats import crossval, nested_crossval, permutation_distribution

n = np.random.randint(3, 5)
stimulus, response, fs = load_sample_data(n_segments=n)


def test_nested_crossval():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    splits = np.random.randint(2, 5)
    metric, best_regularization = nested_crossval(
        trf, stimulus, response, fs, tmin, tmax, reg, splits
    )


def test_pearsonr():
    vec1 = np.random.uniform(0, 1000, 100)
    vec2 = np.random.uniform(0, 1000, 100)
    r1 = np.corrcoef(vec1, vec2)[0, 1]
    r2 = stats.pearsonr(vec1, vec2)
    assert np.allclose(r1, r2)


def test_crossval():
    for direction in [1, -1]:
        tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
        reg = np.random.uniform(0, 10)
        trf = TRF(direction=direction)
        splits = np.random.randint(2, n + 1)
        metric = crossval(trf, stimulus, response, fs, tmin, tmax, reg, splits)
        assert np.isscalar(metric)
        metric = crossval(
            trf, stimulus, response, fs, tmin, tmax, reg, splits, average=False
        )
        if direction == 1:
            assert len(metric) == response[0].shape[-1]
        else:
            assert len(metric) == stimulus[0].shape[-1]


def test_permutation():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(2, 5)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    metric = permutation_distribution(
        trf,
        stimulus,
        response,
        fs,
        tmin,
        tmax,
        reg,
        n_permute,
        k=-1,
    )
    assert len(metric) == n_permute


def test_permutation_average_false():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(5, 10)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    n_channels = response[0].shape[-1]
    metric = permutation_distribution(
        trf, stimulus, response, fs, tmin, tmax, reg, n_permute, average=False
    )
    assert metric.shape == (n_permute, n_channels)


def test_permutation_too_many():
    trf = TRF()
    n_permute = math.factorial(n) + 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        permutation_distribution(
            trf,
            stimulus,
            response,
            fs,
            tmin=0,
            tmax=0.1,
            regularization=1,
            n_permute=n_permute,
        )
    assert len(w) == 1
    assert "will be repeated" in str(w[0].message)
