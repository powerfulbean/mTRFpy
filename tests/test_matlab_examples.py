from pathlib import Path
import numpy as np
from scipy.io import loadmat
from mtrf.model import TRF

root = Path(__file__).parent.absolute()


def test_encoding():
    # load the data
    speech_response = loadmat(str(root / "data" / "speech_data.mat"))
    fs = speech_response["fs"][0][0]
    response = speech_response["resp"]
    stimuli = speech_response["stim"]

    # and the expected result
    encoder_results = loadmat(str(root / "results" / "encoder_results.mat"))
    # w = input features (stimuli) x times x output features (=channels)
    w, b, times, _, direction, kind = encoder_results["modelEncoder"][0][0]
    prediction1 = encoder_results["predResp"]
    correlation1 = encoder_results["predRespStats"]["r"][0][0][0]
    error1 = encoder_results["predRespStats"]["err"][0][0][0]
    # train the TRF model on the data
    trf_encoder = TRF()
    tmin, tmax = -0.1, 0.2
    trf_encoder.train(stimuli, response, fs, tmin, tmax, 100)
    # use the trained TRF to predict data
    prediction2, correlation2, error2 = trf_encoder.predict(
        stimuli, response, average_features=False
    )

    # check that the results are the same as in matlab
    np.testing.assert_almost_equal(trf_encoder.weights, w, decimal=12)
    np.testing.assert_almost_equal(trf_encoder.bias, b, decimal=12)
    np.testing.assert_equal(trf_encoder.times, times[0] / 1e3)
    np.testing.assert_almost_equal(prediction1, prediction2, decimal=12)
    np.testing.assert_almost_equal(correlation1, correlation2, decimal=12)
    np.testing.assert_almost_equal(error1, error2, decimal=12)

    # we should get the same results if we duplicate the data and use the
    # fit function
    stimuli = np.stack([stimuli for _ in range(10)], axis=0)
    response = np.stack([response for _ in range(10)], axis=0)
    trf_encoder = TRF()
    tmin, tmax = -0.1, 0.2
    trf_encoder.fit(stimuli, response, fs, tmin, tmax, 100)
    prediction2, correlation2, error2 = trf_encoder.predict(
        stimuli, response, average_features=False
    )

    # check that the results are the same as in matlab
    np.testing.assert_almost_equal(trf_encoder.weights, w, decimal=12)
    np.testing.assert_almost_equal(trf_encoder.bias, b, decimal=12)
    np.testing.assert_equal(trf_encoder.times, times[0] / 1e3)
    np.testing.assert_almost_equal(correlation1, correlation2, decimal=12)
    np.testing.assert_almost_equal(error1, error2, decimal=12)


def test_decoding():
    # load data and expected results
    speech_response = loadmat(str(root / "data" / "speech_data.mat"))
    fs = speech_response["fs"][0][0]
    response = speech_response["resp"]
    stimuli = speech_response["stim"]
    decoder_results = loadmat(str(root / "results" / "decoder_results.mat"))
    w, b, times, _, direction, kind = decoder_results["modelDecoder"][0][0]
    prediction1 = decoder_results["predStim"]
    correlation1 = decoder_results["predStimStats"]["r"][0][0][0]
    error1 = decoder_results["predStimStats"]["err"][0][0][0]
    # train the model and predict stimulus
    trf_decoder = TRF(direction=-1)
    tmin, tmax = -0.1, 0.2
    trf_decoder.train(stimuli, response, fs, tmin, tmax, 100)
    prediction2, correlation2, error2 = trf_decoder.predict(
        stimuli, response, average_features=False
    )
    # check that the results are the same as in matlab
    np.testing.assert_almost_equal(trf_decoder.weights, w, decimal=11)
    np.testing.assert_almost_equal(trf_decoder.bias, b, decimal=11)
    np.testing.assert_equal(trf_decoder.times, times[0] / 1e3)
    np.testing.assert_almost_equal(prediction1, prediction2, decimal=11)
    np.testing.assert_almost_equal(correlation1, correlation2, decimal=11)
    np.testing.assert_almost_equal(error1, error2, decimal=11)
