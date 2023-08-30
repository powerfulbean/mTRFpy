import numpy as np
from mtrf.model import TRF, load_sample_data
from mtrf.stats import cross_validate, permutation_distribution

n = np.random.randint(5, 10)
stimulus, response, fs = load_sample_data(n_segments=n)


def test_crossval():
    for direction in [1, -1]:
        tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
        reg = np.random.uniform(0, 10)
        trf = TRF(direction=direction)
        splits = np.random.randint(2, 5)
        loss = cross_validate(trf, stimulus, response, fs, tmin, tmax, reg, splits)
        assert np.isscalar(loss)
        loss = cross_validate(
            trf, stimulus, response, fs, tmin, tmax, reg, splits, average=False
        )
        if direction == 1:
            assert len(loss) == response[0].shape[-1]
        else:
            assert len(loss) == stimulus[0].shape[-1]


def test_permutation():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(5, 100)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    loss = permutation_distribution(
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
    assert len(loss) == n_permute
