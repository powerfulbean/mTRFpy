from pathlib import Path
import tempfile
import numpy as np
from numpy.random import randint
from mtrf.model import TRF, cross_validate

root = Path(__file__).parent.absolute()

speech_response = np.load(root / "data" / "speech_data.npy", allow_pickle=True).item()
fs = speech_response["samplerate"][0][0]
response = speech_response["response"]
stimulus = speech_response["stimulus"]


def test_train():
    reps = np.random.randint(2, 10)
    stimuli = [stimulus[0:100] for _ in range(reps)]
    responses = [response[0:100] for _ in range(reps)]
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    regularization = np.random.uniform(0, 10)
    trf1 = TRF(direction=direction)
    trf1.train(stimuli, responses, fs, tmin, tmax, regularization)
    trf2 = TRF(direction=direction)
    trf2.train(stimuli[0], responses[0], fs, tmin, tmax, regularization)
    if direction == 1:
        assert trf1.weights.shape[0] == stimulus.shape[-1]
        assert trf1.weights.shape[-1] == response.shape[-1]
    if direction == -1:
        assert trf1.weights.shape[-1] == stimulus.shape[-1]
        assert trf1.weights.shape[0] == response.shape[-1]
    np.testing.assert_almost_equal(trf1.weights, trf2.weights, 9)


def test_predict():
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    regularization = np.random.uniform(0, 10)
    trf = TRF()
    trf.train(stimulus, response, fs, tmin, tmax, regularization)
    reps = np.random.randint(2, 10)
    stimuli = [stimulus for _ in range(reps)]
    responses = [response for _ in range(reps)]
    predictions, correlations, errors = trf.predict(stimuli, responses)
    assert len(predictions) == len(responses)
    assert all([p[0].shape == r[0].shape for p, r in zip(predictions, responses)])
    assert np.isscalar(correlations) and np.isscalar(errors)
    predictions, correlations, error = trf.predict(stimuli, responses, average=False)
    assert correlations.shape[-1] == trf.weights.shape[-1]


def test_crossval():
    reps = np.random.randint(5, 10)
    responses = [response for _ in range(reps)]
    stimuli = [stimulus for _ in range(reps)]
    tmin = np.random.uniform(-0.1, 0.05)
    tmax = np.random.uniform(0.1, 0.4)
    direction = np.random.choice([1, -1])
    reg = np.random.uniform(0, 10)
    trf = TRF(direction=direction)
    splits = np.random.randint(2, 10)
    test_size = np.random.uniform(0.05, 0.3)
    correlations, errors = cross_validate(
        trf, stimuli, responses, fs, tmin, tmax, reg, splits
    )
    assert np.isscalar(correlations) and np.isscalar(errors)
    correlations, errors = cross_validate(
        trf, stimuli, responses, fs, tmin, tmax, reg, splits, average=False
    )


def test_fit():
    reps = np.random.randint(5, 10)
    responses = [response for _ in range(reps)]
    stimuli = [stimulus for _ in range(reps)]
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
