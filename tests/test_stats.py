from pathlib import Path
import tempfile
import numpy as np
from numpy.random import randint
from mtrf.model import TRF, load_sample_data
from mtrf.stats import cross_validate, permutation_distribution

n = np.random.randint(5, 10)
stimulus, response, fs = load_sample_data(n_segments=n)


def test_crossval():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    reg = np.random.uniform(0, 10)
    trf = TRF(direction=direction)
    splits = np.random.randint(2, 5)
    test_size = np.random.uniform(0.05, 0.3)
    r, mse = cross_validate(trf, stimulus, response, fs, tmin, tmax, reg, splits)
    assert np.isscalar(r) and np.isscalar(mse)
    r, mse = cross_validate(
        trf, stimulus, response, fs, tmin, tmax, reg, splits, average=False
    )
    if direction == 1:
        assert len(r) == len(mse) == response[0].shape[-1]
    else:
        assert len(r) == len(mse) == stimulus[0].shape[-1]


def test_permutation():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(5, 100)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    r, mse = permutation_distribution(
        trf,
        stimulus,
        response,
        fs,
        tmin,
        tmax,
        reg,
        n_permute,
        k=-1,
        average=[1, 2, 3],
    )
    assert len(r) == len(mse) == n_permute
