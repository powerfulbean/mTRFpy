from pathlib import Path
from scipy.io import loadmat
from mtrf.model import TRF

root = Path(__file__).parent.absolute()


def test_plotting():
    # load the data
    speech_response = loadmat(str(root / "data" / "speech_data.mat"))
    fs = speech_response["fs"][0][0]
    response = speech_response["resp"]
    stimuli = speech_response["stim"]
    trf_encoder = TRF()
    tmin, tmax = -0.1, 0.4
    trf_encoder.train(
        stimuli, response * speech_response["factor"][0][0], fs, tmin, tmax, 0.5
    )
    trf_encoder.plot_forward_weights(channels=85, kind="image", show=False)
    trf_encoder.plot_forward_weights(
        channels=[4, 7, 10], tmin=0.1, tmax=0.2, kind="line", show=False
    )
    trf_encoder.plot_forward_weights(kind="image", mode="gfp", show=False)
