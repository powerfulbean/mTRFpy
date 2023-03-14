from pathlib import Path
from itertools import product
import pickle
import requests
from collections.abc import Iterable
import numpy as np
from matplotlib import pyplot as plt
from mtrf.crossval import cross_validate, _progressbar
from mtrf.matrices import (
    covariance_matrices,
    banded_regularization_coefficients,
    regularization_matrix,
    lag_matrix,
    truncate,
    _check_data,
)

try:
    import mne
except:
    mne = None


class TRF:
    """
    Temporal response function.

    A (multivariate) regression using time lagged input features which can be used as
    an encoding model to predict brain responses from stimulus features or as a decoding
    model to predict stimulus features from brain responses.

    Parameters
    ----------
    direction: int
        If 1, make a forward model using the stimulus as predictor to estimate the
        response (default). If -1, make a decoding model using the response to estimate
        the stimulus.
    kind: str
        If 'multi' (default), fit a multi-lag model using all time lags simultaneously.
        If 'single', fit separate single-lag models for each individual lag.
    zeropad: bool
        If True (defaul), pad the outer rows of the design matrix with zeros.
        If False, delete them.
    method: str
        Regularization method. Can be 'ridge' (default), 'tikhonov' or 'banded'.
        See documentation for a detailed explanation.

    Attributes
    ----------
    weights: numpy.ndarray
        Beta coefficients in a three-dimensional inputs-by-lags-by-outputs matrix
    bias: numpy.ndarray
        One dimensional array with one bias term per input feature
    times: list
        Time lags, depending on training time window and sampling rate.
    """

    def __init__(
        self, direction=1, kind="multi", zeropad=True, bias=True, method="ridge"
    ):
        self.weights = None
        self.bias = bias
        self.times = None
        self.fs = None
        self.regularization = None
        if direction in [1, -1]:
            self.direction = direction
        else:
            raise ValueError("Parameter direction must be either 1 or -1!")
        if kind in ["multi", "single"]:
            self.kind = kind
        else:
            raise ValueError('Paramter kind must be either "multi" or "single"!')
        if isinstance(zeropad, bool):
            self.zeropad = zeropad
        else:
            raise ValueError("Parameter zeropad must be boolean!")
        if method in ["ridge", "tikhonov", "banded"]:
            self.method = method
        else:
            raise ValueError('Method must be either "ridge", "tikhonov" or "banded"!')

    def __radd__(self, trf):
        if trf == 0:
            return self.copy()
        else:
            return self.__add__(trf)

    def __add__(self, trf):
        if not isinstance(trf, TRF):
            raise TypeError("Can only add to another TRF instance!")
        if not (self.direction == trf.direction) and (self.kind == trf.kind):
            raise ValueError("Added TRFs must be of same kind and direction!")
        trf_new = self.copy()
        trf_new.weights += trf.weights
        trf_new.bias += trf.bias
        return trf_new

    def __truediv__(self, num):
        trf_new = self.copy()
        trf_new.weights /= num
        trf_new.bias /= num
        return trf_new

    def fit(
        self,
        stimulus,
        response,
        fs,
        tmin,
        tmax,
        regularization,
        bands=None,
        k=5,
        seed=None,
        verbose=True,
    ):
        """
        Train TRF model and fit regularization parameter to data.

        Compute a linear mapping between stimulus and response, or vice versa, using
        regularized regression. This method can compare multiple values for the
        regularization parameter and select the best one (i.e. the one that yields
        the highest prediction accuracy).

        Parameters
        ----------
        stimulus: list
            Each element must contain one trial's stimulus in a two-dimensional
            samples-by-features array (second dimension can be omitted if there is
            only a single feature.
        response: list
            Each element must contain one trial's response in a two-dimensional
            samples-by-features array.
        fs: int
            Sample rate of stimulus and response in hertz.
        tmin: float
            Minimum time lag in seconds.
        tmax: float
            Maximum time lag in seconds.
        regularization: list or float or int
            Lambda parameter for regularization. If a list is supplied, the model is
            fitted separately for each value and the one yielding the highest accuracy
            is chosen (correlation and mean squared error of each model are returned).
        bands: list or None
            Must only be provided when using banded ridge regression. Size of the
            features for which a regularization parameter is fitted, in the order they
            appear in the stimulus matrix. For example, when the stimulus consists of
            an envelope vector and a 16-band spectrogram, bands would be [1, 16].
        k: int
            Number of data splits for cross validation, defaults to 5.
            If -1, do leave-one-out cross-validation.
        seed: int
            Seed for the random number generator.

        Returns
        -------
        correlation: list
            Correlation of prediction and actual output.
        error: list
            Mean squared error between prediction and actual output.
        """
        if not (isinstance(stimulus, list) and isinstance(response, list)):
            raise ValueError(
                "Model fitting requires a list of multiple trials for stimulus and response!"
            )
        else:
            stimulus, response = _check_data(stimulus), _check_data(response)

        if self.method == "banded":
            if bands is None:
                raise ValueError(
                    "Must provide band sizes when using banded ridge regression!"
                )
            if not sum(bands) == stimulus[0].shape[-1]:
                raise ValueError(
                    "Sum of the bands must match the total number of stimulus features!"
                )
            n_lags = int(np.ceil(tmax * fs) - np.floor(tmin * fs) + 1)
            coefficients = list(product(regularization, repeat=2))
            regularization = [
                banded_regularization_coefficients(n_lags, c, bands, self.bias)
                for c in coefficients
            ]
        if np.isscalar(regularization):
            self.train(stimulus, response, fs, tmin, tmax, regularization)
        else:  # run cross-validation once per regularization parameter
            correlation = np.zeros(len(regularization))
            error = np.zeros(len(regularization))
            for ir in _progressbar(
                range(len(regularization)), "Hyperparameter optimization"
            ):
                r = regularization[ir]
                reg_correlation, reg_error = cross_validate(
                    self.copy(), stimulus, response, fs, tmin, tmax, r, k, seed=seed
                )
                correlation[ir] = reg_correlation
                error[ir] = reg_error
            regularization = list(regularization)[np.argmax(correlation)]
            self.train(stimulus, response, fs, tmin, tmax, regularization)
            return correlation, error

    def train(self, stimulus, response, fs, tmin, tmax, regularization):
        """
        Compute the linear mapping between stimulus and response.

        By de-convolution of the stimulus and response we obtain a matrix of weights
        (temporal response function) which is applied to the stimulus via convolution
        to predict the response (or vice versa) so that the mean squared error between
        the prediction and actual output is minimized.

        Parameters
        ----------
        stimulus: list or numpy.ndarray
            Either a 2-D samples-by-features array, if the data contains only one trial
            or a list of such arrays of it contains multiple trials. The second
            dimension can be omitted if there only is a single stimulus feature.
        response: list or numpy.ndarray
            Either a 2-D samples-by-channels array, if the data contains only one trial
            or a list of such arrays of it contains multiple trials.
        fs: int
            Sample rate of stimulus and response in hertz.
        tmin: float
            Minimum time lag in seconds
        tmax: float
            Maximum time lag in seconds
        regularization: float or int
            The regularization parameter (lambda).
        """
        if isinstance(self.bias, np.ndarray):  # reset bias if trf is already trained
            self.bias = True
        stimulus, response = _check_data(stimulus), _check_data(response)
        if not len(stimulus) == len(response):
            ValueError("Response and stimulus must have the same length!")
        else:
            ntrials = len(stimulus)
        if isinstance(regularization, np.ndarray):  # check if matrix is diagonal
            if (
                np.count_nonzero(regularization - np.diag(np.diagonal(regularization)))
                > 0
            ):
                raise ValueError(
                    "Regularization parameter must be a single number or a diagonal matrix!"
                )
        delta = 1 / fs
        self.fs = fs
        self.regularization = regularization
        cov_xx = 0
        cov_xy = 0
        if self.direction == 1:
            xs, ys = stimulus, response
        elif self.direction == -1:
            xs, ys = response, stimulus
            tmin, tmax = -1 * tmax, -1 * tmin
        else:
            raise ValueError(f"trf direction value: {self.direction} is not supported.")
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        for x, y in zip(xs, ys):
            assert x.ndim == 2 and y.ndim == 2
            # sum covariances matrices across observations
            cov_xx_trial, cov_xy_trial = covariance_matrices(
                x, y, lags, self.zeropad, self.bias
            )
            cov_xx += cov_xx_trial
            cov_xy += cov_xy_trial
        cov_xx, cov_xy = cov_xx / ntrials, cov_xy / ntrials  # normalize
        regmat = regularization_matrix(cov_xx.shape[1], self.method)
        regmat *= regularization / delta
        # calculate reverse correlation:
        weight_matrix = np.matmul(np.linalg.inv(cov_xx + regmat), cov_xy) / delta
        self.bias = weight_matrix[0:1]
        self.weights = weight_matrix[1:].reshape(
            (x.shape[1], len(lags), y.shape[1]), order="F"
        )
        self.times = np.array(lags) / fs
        self.fs = fs

    def predict(
        self,
        stimulus=None,
        response=None,
        lag=None,
        average=True,
    ):
        """
        Predict response from stimulus (or vice versa) using the trained model.

        The TRF is convolved with the input to predict the output. If the output is
        provided, this method will estimate the correlation and mean squared error of
        the prediction and actual output.

        Parameters
        ----------
        stimulus: None or list or numpy.ndarray
            Either a 2-D samples-by-features array, if the data contains only one trial
            or a list of such arrays of it contains multiple trials. The second
            dimension can be omitted if there only is a single stimulus feature
            (e.g. envelope). When using a forward model, this must be specified.
        response: None or list or numpy.ndarray
            Either a 2-D samples-by-channels array, if the data contains only one
            trial or a list of such arrays of it contains multiple trials. Must be
            provided when using a backward model.
        lag: None or in or list
            If not None (default), only use the specified lags for prediction.
            The provided values index the elements in self.times.
        average: bool
            If True (default), correlation and mean squared error are averaged
            across all predicted features.

        Returns # continue here
        -------
        prediction: np.ndarray
        Predicted output. Has the same shape as the input size of the last dimension (i.e. features) is equal to the last dimension in self.weights.
        correlation (float, np.ndarray): Scalar if average is True, 1-dimensioal array otherwise.
        error (float, np.ndarray):Scalar if average is True, 1-dimensional array otherwise.
        """
        # check that inputs are valid
        if self.weights is None:
            raise ValueError("Can't make predictions with an untrained model!")
        if self.direction == 1 and stimulus is None:
            raise ValueError("Need stimulus to predict with a forward model!")
        elif self.direction == -1 and response is None:
            raise ValueError("Need response to predict with a backward model!")
        if stimulus is not None:
            stimulus = _check_data(stimulus)
            ntrials = len(stimulus)
        if response is not None:
            response = _check_data(response)
            ntrials = len(stimulus)
        if stimulus is None:
            stimulus = [None for _ in range(ntrials)]
        if response is None:
            response = [None for _ in range(ntrials)]

        xs, ys = None, None
        if self.direction == 1:
            xs, ys = stimulus, response
        elif self.direction == -1:
            xs, ys = response, stimulus
        else:
            raise ValueError(f"trf direction value: {self.direction} is not supported.")

        prediction, correlation, error = [], [], []  # output lists
        for x, y in zip(xs, ys):
            x_samples, x_features = x.shape
            if y is None:
                y_samples = x_samples
                y_features = self.weights.shape[-1]
            else:
                y_samples, y_features = y.shape

            lags = list(
                range(
                    int(np.floor(self.times[0] * self.fs)),
                    int(np.ceil(self.times[-1] * self.fs)) + 1,
                )
            )
            delta = 1 / self.fs

            w = self.weights.copy()
            if lag is not None:  # select lag and corresponding weights
                if not isinstance(lag, Iterable):
                    lag = [lag]
                lags = list(np.array(lags)[lag])
                w = w[:, lag, :]
            w = (
                np.concatenate(
                    [
                        self.bias,
                        w.reshape(x_features * len(lags), y_features, order="F"),
                    ]
                )
                * delta
            )
            x_lag = lag_matrix(x, lags, self.zeropad)
            y_pred = x_lag @ w
            if y is not None:
                if self.zeropad is False:
                    y = truncate(y, lags[0], lags[-1])
                err = np.mean((y - y_pred) ** 2, axis=0)
                r = np.mean((y - y.mean(0)) * (y_pred - y_pred.mean(0)), 0) / (
                    y.std(0) * y_pred.std(0)
                )
                correlation.append(r)
                error.append(err)
            prediction.append(y_pred)
        if ys is not None:
            if average is True:
                correlation, error = np.mean(correlation), np.mean(error)
            else:  # only average across trials, not across channels/features
                correlation, error = np.mean(correlation, axis=0), np.mean(
                    error, axis=0
                )
            return prediction, correlation, error
        else:
            return prediction

    def save(self, path):
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"The directory {path.parent} does not exist!")
        with open(path, "wb") as fname:
            pickle.dump(self, fname, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist!")
        with open(path, "rb") as fname:
            trf = pickle.load(fname)
        self.__dict__ = trf.__dict__

    def copy(self):
        trf = TRF()
        for k, v in self.__dict__.items():
            value = v
            if getattr(v, "copy", None) is not None:
                value = v.copy()
            setattr(trf, k, value)
        return trf

    def plot(
        self,
        channel=None,
        feature=None,
        axes=None,
        show=True,
        kind="line",
    ):
        """
        Plot the weights of the (forward) model across time for a select channel or feature.

        Arguments:
            channel (None | int | str): Channel selection. If None, all channels will be used. If an integer, the channel at that index will be used. If 'avg' or 'gfp' , the average or standard deviation across channels will be computed.
            feature (None | int | str): Feature selection. If None, all features will be used. If an integer, the feature at that index will be used. If 'avg' , the average across features will be computed.
            axes (matplotlib.axes.Axes): Axis to plot to. If None is provided (default) generate a new plot.
            show (bool): If True (default), show the plot after drawing.
            kind (str): Type of plot to draw. If 'line' (default), average the weights across all stimulus features, if 'image' draw a features-by-times plot where the weights are color-coded.

        Returns:
            fig (matplotlib.figure.Figure): If now axes was provided and a new figure is created, it is returned.
        """
        if self.direction == -1:
            # TODO: implement backward to forward transformation
            raise ValueError("Not implemented for decoding models!")
        if axes is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = None, axes  # dont create a new figure
        weights = self.weights
        # select channel and or feature
        if channel is None and feature is None:
            raise ValueError("You must specify a subset of channels or features!")
        if feature is not None:
            image_ylabel = "channel"
            if isinstance(feature, int):
                weights = weights[feature, :, :]
            elif feature == "avg":
                weights = weights.mean(axis=0)
            else:
                raise ValueError('Argument `feature` must be an integer or "avg"!')
        if channel is not None:
            image_ylabel = "feature"
            if isinstance(channel, int):
                weights = weights.T[channel].T
            elif channel == "avg":
                weights = weights.mean(axis=-1)
            elif channel == "gfp":
                weights = weights.std(axis=-1)
            else:
                raise ValueError(
                    'Argument `channel` must be an integer, "avg" or "gfp"'
                )
            weights = weights.T  # transpose so first dimension is time
        # plot the result
        if kind == "line":
            ax.plot(
                self.times.flatten(), weights, linewidth=2 - 0.01 * weights.shape[-1]
            )
            ax.set(
                xlabel="Time lag[s]",
                ylabel="Amplitude [a.u.]",
                xlim=(self.times.min(), self.times.max()),
            )
        elif kind == "image":
            scale = self.times.max() / len(self.times)
            im = ax.imshow(
                weights.T,
                origin="lower",
                aspect="auto",
                extent=[0, weights.shape[0], 0, weights.shape[1]],
            )
            extent = np.asarray(im.get_extent(), dtype=float)
            extent[:2] *= scale
            im.set_extent(extent)
            ax.set(
                xlabel="Time lag [s]",
                ylabel=image_ylabel,
                xlim=(self.times.min(), self.times.max()),
            )
        if show is True:
            plt.show()
        if fig is not None:
            return fig

    @property
    def ftc_weights(self):
        """
        Generate a list of mne.EvokedArray for TRF of different features.

        Returns:
             weights (numpy.ndarray): the TRF weights with the shape always be (n_input_features/channels, n_time_lags, n_output_features/channels).
        """
        if self.direction == -1:
            weights = self.weights.T
        else:
            weights = self.weights
        return weights

    def to_mne_evoked(self, mne_info, **kwargs):
        """
        Generate a list of mne.EvokedArray for TRF of different features.

        Arguments:
            mne_info (mne.Info): the mne.Info object provide necessary information to build the evoked array
            kwargs: other arguments that the user want to feed to the mne.EvokedArray

        Returns:
            single or a list of the constructed evokedArray
        """
        w = [
            mne.EvokedArray(w.T, mne_info, tmin=self.times[0], **kwargs)
            for w in self.ftc_weights
        ]
        return w


def load_sample_data(path=None):
    """
    Load the sample data containing a small snippet of brain responses to naturalistic
    speech and the 16-band spectrogram of that speech. If necessary, the data will be automatically downloaded.

    :param path: full path to the folder where the sample data will be stored. If None (default), a folder called mtrf_data in the users home directory is assumed and created if it does not exist.
    :type path: str or pathlib.Path or None

    :returns

        :path (str | inst of Path | None):     Returns:
        :stimulus (np.ndarray): 16-bands spectrogram of the presented speech.
        :response (np.ndarray): 128-channel EEG recording.
        :fs (int): sampling rate in Hz.
    """
    if path == None:  # use default path
        path = Path.home() / "mtrf_data"
        if not path.exists():
            path.mkdir()
    else:
        path = Path(path)
    if not (path / "speech_data.npy").exists():  # download the data
        url = "https://github.com/powerfulbean/mTRFpy/raw/master/tests/data/speech_data.npy"
        response = requests.get(url, allow_redirects=True)
        open(path / "speech_data.npy", "wb").write(response.content)
    data = np.load(str(path / "speech_data.npy"), allow_pickle=True).item()
    return data["stimulus"], data["response"], data["samplerate"][0][0]


def kwargs_trf_mne_joint(trf=None):
    """
    Returns:
        kwargs (dict): the suggested kwargs for the mne plot_joint funciton
    """
    kwargs = {}
    kwargs["topomap_args"] = {"scalings": 1}
    kwargs["ts_args"] = {"units": "a.u.", "scalings": dict(eeg=1)}
    return kwargs


def kwargs_trf_mne_topo(trf=None):
    """
    Returns:
        kwargs (dict): the suggested kwargs for mne topomap (series)
    """
    kwargs = {
        "cmap": "jet",
        "time_unit": "s",
        "scalings": 1,
        "units": "a.u.",
        "cbar_fmt": "%3.3f",
    }
    return kwargs


def kwargs_r_mne_topo(trf=None):
    """
    Returns:
        kwargs (dict): the suggested kwargs for mne topomap (single)
    """
    kwargs = {
        "times": [0],
        "cmap": "jet",
        "time_unit": "s",
        "scalings": 1,
        "units": "r",
        "cbar_fmt": "%3.3f",
    }
    return kwargs


def foo(a, b):
    """Function

    Args:
        a (float): First number
        b (float): Second number

    Returns:
        result_sum (float): Sum of numbers
        result_prod (float): Product of numbers
    """
    result_sum = a + b
    result_prod = a * b
    return result_sum, result_prod
