from pathlib import Path
from itertools import product
import pickle
from collections.abc import Iterable
import numpy as np
from mtrf.stats import (
    _crossval,
    _progressbar,
    _check_k,
    neg_mse,
    pearsonr,
)
from mtrf.matrices import (
    covariance_matrices,
    banded_regularization,
    regularization_matrix,
    lag_matrix,
    truncate,
    _check_data,
    _get_xy,
)

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    import mne
except ModuleNotFoundError:
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
        If True (default), pad the outer rows of the design matrix with zeros.
        If False, delete them.
    method: str
        Regularization method. Can be 'ridge' (default), 'tikhonov' or 'banded'.
        See documentation for a detailed explanation.
    preload: bool
        If True (default), covariance matrices for all trials will be pre-loaded before
        cross-validation. This makes optimization faster but consumes more memory. If
        False, the covariance matrices will be computed on each iteration which is slower
        but memory efficient.
    metric: callable
        A callable which accept two arguments (true y, predicted y), and retrun a single
        value for each feature in y. The default is mtrf.stats.pearsonr.

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
        self,
        direction=1,
        kind="multi",
        zeropad=True,
        method="ridge",
        preload=True,
        metric=pearsonr,
    ):
        self.weights = None
        self.bias = None
        self.times = None
        self.fs = None
        self.regularization = None
        if not callable(metric):
            raise ValueError("Metric function must be callable")
        else:
            self.metric = metric
        if isinstance(preload, bool):
            self.preload = preload
        else:
            raise ValueError("Parameter preload must be either True or False!")
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

    def train(
        self,
        stimulus,
        response,
        fs,
        tmin,
        tmax,
        regularization,
        bands=None,
        k=-1,
        average=True,
        seed=None,
        verbose=True,
    ):
        """
        Optimize the regularization parameter.

        For each value in `regularization`, a model is trained and validated using
        k-fold cross validation. Then the best regularization value (i.e. the one
        that results in the highest mean metric value is chosen to train the final model.

        Compute a linear mapping between stimulus and response, or vice versa, using
        regularized regression. This method can compare multiple values for the
        regularization parameter and select the best one (i.e. the one that yields
        the highest prediction metric).

        Parameters
        ----------
        stimulus: list
            Each element must contain one trial's stimulus in a two-dimensional
            samples-by-features array (second dimension can be omitted if there is
            only a single feature.
        response: list
            Each element must contain one trial's response in a two-dimensional
            samples-by-channels array.
        fs: int
            Sample rate of stimulus and response in hertz.
        tmin: float
            Minimum time lag in seconds.
        tmax: float
            Maximum time lag in seconds.
        regularization: list or float or int
            Values for the regularization parameter lambda. The model is fitted
            separately for each value and the one yielding the highest accuracy
            is chosen (correlation and mean squared error of each model are returned).
        bands: list or None
            Must only be provided when using banded ridge regression. Size of the
            features for which a regularization parameter is fitted, in the order they
            appear in the stimulus matrix. For example, when the stimulus consists of
            an envelope vector and a 16-band spectrogram, bands would be [1, 16].
        k: int
            Number of data splits for cross validation, defaults to -1 (leave-one-out).
        average: bool or list or numpy.ndarray
            If True (default), average metric cross all predictions (e.g. channels in the
            case of forward modelling). If `average` is an array of indices only average
            those features.
        seed: int
            Seed for the random number generator.
        verbose: bool
            If True (default), show a progress bar during fitting.

        Returns
        -------
        metric : list
            When providing multiple `regularization` values this returns the metric as
            computed by the metric function defined in the `TRF.metric` attribute
            for every regularization value.
        """
        if average is False:
            raise ValueError("Average must be True or a list of indices!")
        stimulus, response, n_trials = _check_data(stimulus, response)
        if not np.isscalar(regularization):
            k = _check_k(k, n_trials)
        x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, self.direction)
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        if self.method == "banded":
            coefficients = list(product(regularization, repeat=len(bands)))
            regularization = [
                banded_regularization(len(lags), c, bands) for c in coefficients
            ]
        if np.isscalar(regularization):
            self._train(x, y, fs, tmin, tmax, regularization)
            return
        else:  # run cross-validation once per regularization parameter
            # pre-compute covariance matrices
            cov_xx, cov_xy = None, None
            if self.preload:
                cov_xx, cov_xy = covariance_matrices(
                    x, y, lags, self.zeropad, self.preload
                )
            else:
                cov_xx, cov_xy = None, None
            metric = np.zeros(len(regularization))
            for ir in _progressbar(
                range(len(regularization)),
                "Hyperparameter optimization",
                verbose=verbose,
            ):
                metric[ir] = _crossval(
                    self.copy(),
                    x,
                    y,
                    cov_xx,
                    cov_xy,
                    lags,
                    fs,
                    regularization[ir],
                    k,
                    seed=seed,
                    average=average,
                    verbose=verbose,
                )
            best_regularization = list(regularization)[np.argmax(metric)]
            self._train(x, y, fs, tmin, tmax, best_regularization)
            return metric

    def _train(self, x, y, fs, tmin, tmax, regularization):
        if isinstance(regularization, np.ndarray):  # check if matrix is diagonal
            if (
                np.count_nonzero(regularization - np.diag(np.diagonal(regularization)))
                > 0
            ):
                raise ValueError(
                    "Regularization parameter must be a single number or a diagonal matrix!"
                )
        self.fs, self.regularization = fs, regularization
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        cov_xx, cov_xy = covariance_matrices(x, y, lags, self.zeropad, preload=False)
        regmat = regularization_matrix(cov_xx.shape[1], self.method)
        regmat *= regularization / (1 / self.fs)
        weight_matrix = np.matmul(np.linalg.inv(cov_xx + regmat), cov_xy) / (
            1 / self.fs
        )
        self.bias = weight_matrix[0:1]
        if self.bias.ndim == 1:  # add empty dimension for single feature models
            self.bias = np.expand_dims(self.bias, axis=0)
        self.weights = weight_matrix[1:].reshape(
            (x[0].shape[1], len(lags), y[0].shape[1]), order="F"
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

        The matrix of TRF weights is multiplied with the time-lagged input to predict
        the output. If the actual output is provided, this method will estimate the
        correlation and mean squared error of the predicted and actual output.

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
        average: bool or list or numpy.ndarray
            If True (default), average metric across all predicted features (e.g. channels
            in the case of forward modelling). If `average` is an array of indices only
            average those features. If `False`, return each predicted feature's metric.

        Returns
        -------
        prediction: numpy.ndarray
            Predicted stimulus or response
        metric: float or numpy.ndarray
            If both stimulus and response are provided, metric is computed by the
            metric function defined in the attribute `TRF.metric`.
            If average is False, an array containing the metric for each feature
            is returned.
        """
        # check that inputs are valid
        if self.weights is None:
            raise ValueError("Can't make predictions with an untrained model!")
        if self.direction == 1 and stimulus is None:
            raise ValueError("Need stimulus to predict with a forward model!")
        elif self.direction == -1 and response is None:
            raise ValueError("Need response to predict with a backward model!")
        else:
            stimulus, response, n_trials = _check_data(stimulus, response)
        if stimulus is None:
            stimulus = [None for _ in range(n_trials)]
        if response is None:
            response = [None for _ in range(n_trials)]

        x, y = _get_xy(stimulus, response, direction=self.direction)
        prediction = [np.zeros((x_i.shape[0], self.weights.shape[-1])) for x_i in x]
        metric = []
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            lags = list(
                range(
                    int(np.floor(self.times[0] * self.fs)),
                    int(np.ceil(self.times[-1] * self.fs)) + 1,
                )
            )
            w = self.weights.copy()
            if lag is not None:  # select lag and corresponding weights
                if not isinstance(lag, Iterable):
                    lag = [lag]
                lags = list(np.array(lags)[lag])
                w = w[:, lag, :]
            w = np.concatenate(
                [
                    self.bias,
                    w.reshape(
                        x_i.shape[-1] * len(lags), self.weights.shape[-1], order="F"
                    ),
                ]
            ) * (1 / self.fs)
            x_lag = lag_matrix(x_i, lags, self.zeropad)
            y_pred = x_lag @ w
            if y_i is not None:
                if self.zeropad is False:
                    y_i = truncate(y_i, lags[0], lags[-1])
                metric.append(self.metric(y_i, y_pred))
            prediction[i][:] = y_pred
        if y[0] is not None:
            metric = np.mean(metric, axis=0)  # average across trials
            if isinstance(average, list) or isinstance(average, np.ndarray):
                metric = metric[average]  # select a subset of predictions
            if average is not False:
                metric = np.mean(metric)
            return prediction, metric
        else:
            return prediction

    def to_forward(self, response):
        """
        Transform a backward to a forward model.

        Use the method described in Haufe et al. 2014 to transform the weights of
        a backward model into coefficients reflecting forward activation patterns
        which have a clearer physiological interpretation.

        Parameters
        ----------
        response: list or numpy.ndarray
            response data which was used to train the backward model as single
            trial in a samples-by-channels array or list of multiple trials.

        Returns
        -------
        trf: model.TRF
            New TRF instance with the transformed forward weights
        """
        assert self.direction == -1

        _, response, n_trials = _check_data(None, response)
        stim_pred = self.predict(response=response)

        Cxx = 0
        Css = 0
        trf = self.copy()
        trf.times = np.asarray([-i for i in reversed(trf.times)])
        trf.direction = 1
        for i in range(n_trials):
            Cxx = Cxx + response[i].T @ response[i]
            Css = Css + stim_pred[i].T @ stim_pred[i]
        nStimChan = trf.weights.shape[-1]
        for i in range(nStimChan):
            trf.weights[..., i] = Cxx @ self.weights[..., i] / Css[i, i]

        trf.weights = np.flip(trf.weights.T, axis=1)
        trf.bias = np.zeros(trf.weights.shape[-1])
        return trf

    def save(self, path):
        """
        Save class instance using the pickle format.

        Parameters
        ----------
        path: str or pathlib.Path
            File destination.
        """
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"The directory {path.parent} does not exist!")
        with open(path, "wb") as fname:
            pickle.dump(self, fname, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """
        Load pickle file - instance variables will be overwritten with file content.

        Parameters
        ----------
        path: str or pathlib.Path
            File destination.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist!")
        with open(path, "rb") as fname:
            trf = pickle.load(fname)
        self.__dict__ = trf.__dict__

    def copy(self):
        """Return a copy of the TRF instance"""
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
        if plt is None:
            raise ModuleNotFoundError("Need matplotlib to plot TRF!")
        if self.direction == -1:
            weights = self.weights.T
            print(
                "WARNING: decoder weights are hard to interpret, consider using the `to_forward()` method"
            )
        if axes is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = None, axes  # dont create a new figure
        weights = self.weights
        # select channel and or feature
        if weights.shape[0] == 1:
            feature = 0
        if weights.shape[-1] == 1:
            channel = 0
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

    def to_mne_evoked(self, info, include=None, **kwargs):
        """
        Output TRF weights as instance(s) of MNE-Python's EvokedArray.

        Create one instance of ``mne.EvokedArray`` for each feature along the first
        (i.e. input) dimension of ``self.weights``. When using a backward model,
        the weights are transposed to obtain one EvokedArray per decoded feature.
        See the MNE-Python documentation for details on the Evoked class.

        Parameters
        ----------
        info: mne.Info or mne.channels.montage.DigMontage
            Either a basic info or montage containing channel locations
            Information neccessary to build the EvokedArray.
        include: None or in or list
            Indices of the stimulus features to include. If None (default),
            create one Evoked object for each feature.
        kwargs: dict
            other parameters for constructing the EvokedArray

        Returns
        -------
        evokeds: list
            One Evoked instance for each included TRF feature.
        """
        if mne is False:
            raise ModuleNotFoundError("To use this function, mne must be installed!")
        if self.direction == -1:
            weights = self.weights.T
        else:
            weights = self.weights
        if isinstance(info, mne.channels.montage.DigMontage):
            kinds = [d["kind"] for d in info.copy().remove_fiducials().dig]
            ch_types = []
            for k in kinds:
                if "eeg" in str(k).lower():
                    ch_types.append("eeg")
                if "mag" in str(k).lower():
                    ch_types.append("mag")
                if "grad" in str(k).lower():
                    ch_types.append("grad")
            mne_info = mne.create_info(info.ch_names, self.fs, ch_types)
        elif isinstance(info, mne.Info):
            mne_info = info
        else:
            raise ValueError
        if isinstance(include, list) or isinstance(include, np.ndarray):
            weights = weights[np.asarray(include), :, :]
        evokeds = []
        for w in weights:
            evoked = mne.EvokedArray(w.T.copy(), mne_info, tmin=self.times[0], **kwargs)
            if isinstance(info, mne.channels.montage.DigMontage):
                evoked.set_montage(info)
            evokeds.append(evoked)
        return evokeds


def load_sample_data(path=None, n_segments=1, normalize=True):
    """
    Load sample of brain responses to naturalistic speech.

    If no path is provided, the data is assumed to be in a folder called mtrf_data
    in the users home directory and will be downloaded and stored there if it can't
    be found. The data contains about 2 minutes of brain responses to naturalistic
    speech, recorded with a 128-channel Biosemi EEG system and the 16-band spectrogram
    of that speech.

    Parameters
    ----------
    path: str or pathlib.Path
        Destination where the sample data is stored or will be downloaded to. If None
        (default), a folder called mtrf_data in the users home directory is assumed
        and created if it does not exist.

    Returns
    -------
    stimulus: numpy.ndarray
        Samples-by-features array of the presented speech's spectrogram.
    response : numpy.ndarray
        Samples-by-channels array of the recorded neural response.
    fs: int
        Sampling rate of stimulus and response in Hz.
    """
    if path is None:  # use default path
        path = Path.home() / "mtrf_data"
        if not path.exists():
            path.mkdir()
    else:
        path = Path(path)
    if not (path / "speech_data.npy").exists():  # download the data
        url = "https://github.com/powerfulbean/mTRFpy/raw/master/tests/data/speech_data.npy"
        import requests

        response = requests.get(url, allow_redirects=True)
        open(path / "speech_data.npy", "wb").write(response.content)
    data = np.load(str(path / "speech_data.npy"), allow_pickle=True).item()
    stimulus, response, fs = (
        data["stimulus"],
        data["response"],
        data["samplerate"][0][0],
    )
    stimulus = np.array_split(stimulus, n_segments)
    response = np.array_split(response, n_segments)
    if normalize:
        for i in range(len(stimulus)):
            stimulus[i] = (stimulus[i] - stimulus[i].mean(axis=0)) / stimulus[i].std(
                axis=0
            )
            response[i] = (response[i] - response[i].mean(axis=0)) / response[i].std(
                axis=0
            )
    return stimulus, response, fs
