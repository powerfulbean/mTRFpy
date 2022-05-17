from pathlib import Path
import numpy as np
from scipy.io import loadmat
from mTRFpy.Model import CTRF
root = Path(__file__).parent.absolute()


def test_encoding():
    # load the data
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = speech_response['resp']
    stimuli = speech_response['stim']

    # and the expected result
    encoder_results = loadmat(str(root/'results'/'encoder_results.mat'))
    # w = input features (stimuli) x times x output features (=channels)
    w, b, times, _, direction, kind = encoder_results['modelEncoder'][0][0]
    prediction1 = encoder_results['predResp'][0]
    correlation1 = encoder_results['predRespStats']['r'][0][0][0]
    error1 = encoder_results['predRespStats']['err'][0][0][0]
    # train the TRF model on the data
    trf_encoder = CTRF()
    tmin, tmax = -100, 200
    trf_encoder.train(stimuli, response, 1, fs, tmin, tmax, 100)
    # use the trained TRF to predict data
    prediction2, correlation2, error2 = trf_encoder.predict(stimuli, response)

    # check that the results are the same as in matlab
    np.testing.assert_almost_equal(trf_encoder.w, w, decimal=12)
    np.testing.assert_almost_equal(trf_encoder.b, b,  decimal=12)
    np.testing.assert_equal(trf_encoder.t, times)
    np.testing.assert_almost_equal(prediction1, prediction2[0], decimal=12)
    np.testing.assert_almost_equal(correlation1, correlation2[0], decimal=12)
    np.testing.assert_almost_equal(error1, error2[0], decimal=12)


def test_decoding():
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    response = speech_response['resp']
    stimuli = speech_response['stim']

    # and the expected result
    decoder_results = loadmat(str(root/'results'/'encoder_results.mat'))
