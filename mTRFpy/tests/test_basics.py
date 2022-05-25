from pathlib import Path
import numpy as np
from scipy.io import loadmat
from mTRFpy.Model import lag_matrix
root = Path(__file__).parent.absolute()


def test_lag_matrix():
    # load the data
    speech_response = loadmat(str(root/'data'/'speech_data.mat'))
    fs = speech_response['fs'][0][0]
    stimuli = speech_response['stim']
    for _ in range(10):
        tmin = np.random.randint(-200, 100)/1e3
        tmax = np.random.randint(150, 500)/1e3
        lags = list(range(int(np.floor(tmin*fs)), int(np.ceil(tmax*fs)) + 1))
        lag_mat = lag_matrix(stimuli, lags, True, True)
        assert lag_mat.shape[0] == stimuli.shape[0]
        assert lag_mat.shape[1] == len(lags)*stimuli.shape[1]+1
        lag_mat = lag_matrix(stimuli, lags, True, False)
        assert lag_mat.shape[0] == stimuli.shape[0]
        assert lag_mat.shape[1] == len(lags)*stimuli.shape[1]
        lag_mat = lag_matrix(stimuli, lags, False, False)
        assert stimuli.shape[0] - lag_mat.shape[0] == len(lags)-1
        assert lag_mat.shape[1] == len(lags)*stimuli.shape[1]
        lag_mat = lag_matrix(stimuli, lags, False, True)
        assert stimuli.shape[0] - lag_mat.shape[0] == len(lags)-1
        assert lag_mat.shape[1] == len(lags)*stimuli.shape[1]+1
