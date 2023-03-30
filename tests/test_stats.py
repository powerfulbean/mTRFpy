from pathlib import Path
import tempfile
import numpy as np
from numpy.random import randint
from mtrf.model import TRF, load_sample_data
from mtrf.stats import cross_validate, permutation_distribution

stimulus, response, fs = load_sample_data()


def test_crossval():
    n = np.random.randint(5, 10)
    stimuli, responses = np.array_split(stimulus, n), np.array_split(response, n)
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    reg = np.random.uniform(0, 10)
    trf = TRF(direction=direction)
    splits = np.random.randint(2, 5)
    test_size = np.random.uniform(0.05, 0.3)
    correlations, errors = cross_validate(
        trf, stimuli, responses, fs, tmin, tmax, reg, splits
    )
    assert np.isscalar(correlations) and np.isscalar(errors)
    correlations, errors = cross_validate(
        trf, stimuli, responses, fs, tmin, tmax, reg, splits, average=False
    )


def test_permutation():
    n = np.random.randint(5, 10)
    stimuli, responses = np.array_split(stimulus, n), np.array_split(response, n)
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(5, 100)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    r, mse = permutation_distribution(
        trf,
        stimuli,
        responses,
        fs,
        tmin,
        tmax,
        reg,
        n_permute,
        k=-1,
        average=[1, 2, 3],
    )
    assert len(r) == len(mse) == n_permute
