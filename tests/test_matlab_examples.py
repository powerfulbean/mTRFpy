from pathlib import Path
import numpy as np
from mtrf.model import TRF
from mtrf.stats import neg_mse, pearsonr

root = Path(__file__).parent.absolute()

speech_response = np.load(root / "data" / "speech_data.npy", allow_pickle=True).item()
fs = speech_response["samplerate"][0][0]
response = speech_response["response"]
stimulus = speech_response["stimulus"]


def test_encoding():
    encoder_results = np.load(  # expected results
        root / "results" / "encoder_results.npy", allow_pickle=True
    ).item()
    w, b, times, _, direction, kind = encoder_results["model"]
    prediction1 = encoder_results["prediction"]
    correlation1 = encoder_results["prediction_r"]
    error1 = encoder_results["prediction_err"]
    trf_encoder = TRF(metric=pearsonr)
    tmin, tmax = -0.1, 0.2
    trf_encoder.train(stimulus, response, fs, tmin, tmax, 100)
    prediction2, correlation2 = trf_encoder.predict(stimulus, response, average=False)
    trf_encoder = TRF(metric=neg_mse)
    trf_encoder.train(stimulus, response, fs, tmin, tmax, 100)
    prediction2, error2 = trf_encoder.predict(stimulus, response, average=False)
    error2 = -error2
    # check that the results are the same as in matlab
    np.testing.assert_almost_equal(trf_encoder.weights, w, decimal=12)
    np.testing.assert_almost_equal(trf_encoder.bias, b, decimal=12)
    np.testing.assert_equal(trf_encoder.times, times[0] / 1e3)
    np.testing.assert_almost_equal(prediction1, np.concatenate(prediction2), decimal=12)
    np.testing.assert_almost_equal(correlation1, correlation2, decimal=12)
    np.testing.assert_almost_equal(error1, error2, decimal=12)


def test_decoding():
    decoder_results = np.load(  # expected results
        root / "results" / "decoder_results.npy", allow_pickle=True
    ).item()
    # w = input features (stimuli) x times x output features (=channels)
    w, b, times, _, direction, kind = decoder_results["model"]
    prediction1 = decoder_results["prediction"]
    correlation1 = decoder_results["prediction_r"]
    error1 = decoder_results["prediction_err"]
    # train the model and predict stimulus
    trf_decoder = TRF(direction=-1)
    tmin, tmax = -0.1, 0.2
    trf_decoder.train(stimulus, response, fs, tmin, tmax, 100)
    prediction2, correlation2 = trf_decoder.predict(stimulus, response, average=False)
    trf_decoder = TRF(direction=-1, metric=neg_mse)
    trf_decoder.train(stimulus, response, fs, tmin, tmax, 100)
    prediction2, error2 = trf_decoder.predict(stimulus, response, average=False)
    error2 = -error2
    # check that the results are the same as in matlab
    np.testing.assert_almost_equal(trf_decoder.weights, w, decimal=11)
    np.testing.assert_almost_equal(trf_decoder.bias, b, decimal=11)
    np.testing.assert_equal(trf_decoder.times, times[0] / 1e3)
    np.testing.assert_almost_equal(prediction1, prediction2[0], decimal=11)
    np.testing.assert_almost_equal(correlation1, correlation2, decimal=11)
    np.testing.assert_almost_equal(error1, error2, decimal=11)


def test_transform():
    transform_results = np.load(  # expected results
        root / "results" / "transform_results.npy", allow_pickle=True
    ).item()

    t = transform_results["t"]
    w = transform_results["w"]
    direction = transform_results["dir"][0, 0]

    trf_decoder = TRF(direction=-1)
    tmin, tmax = -0.1, 0.2
    trf_decoder.train(stimulus, response, fs, tmin, tmax, 100)
    trf_trans_enc = trf_decoder.to_forward(response)

    scale = 1e-5
    np.testing.assert_almost_equal(trf_trans_enc.weights * scale, w * scale, decimal=10)
    np.testing.assert_equal(trf_trans_enc.times, t[0] / 1e3)
    assert trf_trans_enc.direction == direction
