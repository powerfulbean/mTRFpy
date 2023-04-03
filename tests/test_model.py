from pathlib import Path
import tempfile
import numpy as np
from numpy.random import randint
from mtrf.model import TRF, load_sample_data

stimulus, response, fs = load_sample_data()


def test_train():
    n = np.random.randint(2, 10)
    stimuli, responses = np.array_split(stimulus, n), np.array_split(response, n)
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    regularization = np.random.uniform(0, 10)
    trf = TRF(direction=direction)
    trf.train(stimuli, responses, fs, tmin, tmax, regularization)
    if direction == 1:
        assert trf.weights.shape[0] == stimulus.shape[-1]
        assert trf.weights.shape[-1] == response.shape[-1]
    if direction == -1:
        assert trf.weights.shape[-1] == stimulus.shape[-1]
        assert trf.weights.shape[0] == response.shape[-1]


def test_predict():
    n = np.random.randint(2, 10)
    stimuli, responses = np.array_split(stimulus, n), np.array_split(response, n)
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    regularization = np.random.uniform(0, 10)
    trf = TRF()
    trf.train(stimulus, response, fs, tmin, tmax, regularization)
    for average in [True, list(range(randint(responses[0].shape[-1])))]:
        predictions, correlations, errors = trf.predict(
            stimuli, responses, average=average
        )
        assert len(predictions) == len(responses)
        assert all([p[0].shape == r[0].shape for p, r in zip(predictions, responses)])
        assert np.isscalar(correlations) and np.isscalar(errors)
    predictions, correlations, error = trf.predict(stimuli, responses, average=False)
    assert correlations.shape[-1] == trf.weights.shape[-1]


def test_fit():
    n = np.random.randint(5, 10)
    stimuli, responses = np.array_split(stimulus, n), np.array_split(response, n)
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    reg = np.random.uniform(0, 10)
    trf = TRF(direction=direction)
    trf.fit(stimuli, responses, fs, tmin, tmax, reg)
    reg = [np.random.uniform(0, 10) for _ in range(randint(2, 10))]
    trf = TRF(direction=direction)
    trf.fit(stimuli, responses, fs, tmin, tmax, reg)


def test_save_load():
    tmpdir = Path(tempfile.gettempdir())
    reps = np.random.randint(2, 10)
    stimuli = [stimulus[0:100] for _ in range(reps)]
    responses = [response[0:100] for _ in range(reps)]
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    regularization = np.random.uniform(0, 10)
    trf1 = TRF(direction=direction)
    trf1.train(stimuli, responses, fs, tmin, tmax, regularization)
    trf1.save(tmpdir / "test.trf")
    trf2 = TRF()
    trf2.load(tmpdir / "test.trf")
    np.testing.assert_equal(trf1.weights, trf2.weights)
