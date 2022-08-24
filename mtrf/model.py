# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""
from pathlib import Path
import pickle
from collections.abc import Iterable
import numpy as np
from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = False


def cross_validate(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    k=5,
    seed=None,
    average_features=True,
    average_splits=True,
):
    """
    Train and test a model using k-fold cross-validation. The input data
    is randomly shuffled and separated into k equally large parts, k-1
    parts are used for training and the kth part is used for testing the model.
    If the data can't be divided evenly among splits, some splits will have one
    trial more as to not waste any data.
    Arguments:
        model (instance of TRF): The model that is fit to the data.
            For every iteration of cross-validation a new copy is created.
        stimulus (np.ndarray | None): Stimulus matrix of shape
            trials x samples x features.
        response (np.ndarray | None):  Response matrix of shape
            trials x samples x features.
        fs (int): Sample rate of stimulus and response in hertz.
        tmin (float): Minimum time lag in seconds
        tmax (float): Maximum time lag in seconds
        regularization (float | int): The regularization paramter (lambda).
        k (int): Number of data splits for cross validation.
                     If -1, do leave-one-out cross-validation.
        seed (int): Seed for the random number generator.
        average_features (bool): If True (default), average correlations
            and scores across all predicted features. Else, return one
            value for every feature.
        average_splits (bool): If true, average models, the correlations and
            erros across all cross-validation splits. Else, return one
            value for each split.
    Returns:
        models (list | instance of TRF): The fitted models(s). If
            average_splits is False, a list with one model for each split
            is returned. Else, the weights are averaged across splits
            and a single model is retruned (default).
        correlations (np.ndarray | float): Correlation between the actual
            and predicted output. If average_features or average_splits
            is False, a separate value for each feature / split is returned
        errors (np.array | float): Mean squared error between the actual
            and predicted output. If average_features or average_splits
            is False, a separate value for each feature / split is returned
    """
    if seed is not None:
        np.random.seed(seed)
    if not stimulus.ndim == 3 and response.ndim == 3:
        raise ValueError("Arrays must be 3D with" "observations x samples x features!")
    if stimulus.shape[0:2] != response.shape[0:2]:
        raise ValueError(
            "Stimulus and response must have same number of" "samples and observations!"
        )
    observations = np.arange(stimulus.shape[0])
    np.random.shuffle(observations)  # randomize trial indices
    if k == -1:  # do leave-one-out cross validation
        splits_test = observations
        splits_train = splits_test[1:] - (splits_test[:, None] >= splits_test[1:])
        n_splits = len(splits_test)
    else:
        splits = np.array_split(observations, k)
        n_splits = len(splits)
    if average_features is True:
        errors, correlations = np.zeros(n_splits), np.zeros(n_splits)
    else:
        if model.direction == 1:
            n_features = response.shape[-1]
        elif model.direction == -1:
            n_features = stimulus.shape[-1]
        correlations = np.zeros((n_splits, n_features))
        errors = np.zeros((n_splits, n_features))
    for fold in range(k):
        if k == -1:
            idx_test, idx_train = splits_test[fold], splits_train[fold]
        else:
            idx_test = splits[fold]
            idx_train = np.concatenate(splits[:fold] + splits[fold + 1 :])
        trf = model.copy()
        trf.train(
            stimulus[idx_train], response[idx_train], fs, tmin, tmax, regularization
        )
        _, fold_correlation, fold_error = trf.predict(
            stimulus[idx_test], response[idx_test], average_features=average_features
        )
        correlations[fold], errors[fold] = fold_correlation, fold_error
    if average_splits:
        correlations, errors = correlations.mean(0), errors.mean(0)
    return correlations, errors


class TRF:
    """
    Class for the (multivariate) temporal response function.
    Can be used as a forward encoding model (stimulus to neural response)
    or backward decoding model (neural response to stimulus) using time lagged
    input features as per Crosse et al. (2016).
    Arguments:
        direction (int): Direction of the model. Can be 1 to fit a forward
            model (default) or -1 to fit a backward model.
        kind (str): Kind of model to fit. Can be 'multi' (default) to fit
            a multi-lag model using all time lags simulatneously or
            'single' to fit separate sigle-lag models for each individual lag.
        zeropad (bool): If True (defaul), pad the outer rows of the design
            matrix with zeros. If False, delete them.
        method (str): Regularization method. Can be 'ridge' (default) or
            'tikhonov'.
    Attributes:
       weights (np.ndarray): Model weights which are estimated by fitting the
           model to the data using the train() method. The weight matrix should
           have the shape stimulus features x time lags x response features.
        bias (np.ndarray): Vector containing the bias term for every
            response feature.
        times (list): Model time lags, estimated based on the training time
            window and sampling rate.

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
        if method in ["ridge", "tikhonov"]:
            self.method = method
        else:
            raise ValueError('Method must be either "ridge" or "tikhonov"!')

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
        k=5,
        seed=None,
        verbose=True,
    ):
        """
        Fit TRF model. If a regularization is just a single scalar, this method
        will simply call `TRF.train`, when given a list of regularization values,
        this method will find the best value (i.e. the one that yields the
        highest prediction accuracy) and train a TRF with the selected regularization value.
        Arguments:
            stimulus (np.ndarray | None): Stimulus matrix of shape
                trials x samples x features.
            response (np.ndarray | None):  Response matrix of shape
                trials x samples x features.
            fs (int): Sample rate of stimulus and response in hertz.
            tmin (float): Minimum time lag in seconds
            tmax (float): Maximum time lag in seconds
            regularization (list, float, int): The regularization paramter
                (lambda). If a list with multiple values is supplied, the
                model is fitted separately for each value. The model with the
                highest accuracy (correlation of prediction and actual output)
                is selected and the correlation and error for every tested
                regularization value are returned.
            k (int): Number of data splits for cross validation.
                     If -1, do leave-one-out cross-validation.
            seed (int): Seed for the random number generator.
        Returns:
            correlation (list): Correlation of prediction and actual output
                per value when using multiple regularization values.
            error (list): Error between prediction and output per value
                when using multiple regularization values.
        """

        if not stimulus.ndim == 3 and response.ndim == 3:
            raise ValueError(
                "TRF fitting requires 3-dimensional arrays"
                "for stimulus and response with the shape"
                "n_stimuli x n_sammples x n_features."
            )
        if np.isscalar(regularization):
            self.train(stimulus, response, fs, tmin, tmax, regularization)
        else:  # run cross-validation once per regularization parameter
            correlation, error = [], []
            if (tqdm is not False) and (verbose is True):
                regularization = tqdm(
                    regularization, leave=False, desc="fitting regularization parameter"
                )
            correlation = np.zeros(len(regularization))
            error = np.zeros(len(regularization))
            for ir, r in enumerate(regularization):
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
        Compute the TRF weights that minimze the mean squared error between the
        actual and predicted neural response.
        Arguments:
            stimulus (np.ndarray): Stimulus data, has to be of shape
                samples x features.
            response (np.ndarray): Neural response, must be of shape
                samples x fetures. Must have the same number of samples
                as the stimulus.
            fs (int): Sample rate of stimulus and response in hertz.
            tmin (float): Minimum time lag in seconds
            tmax (float): Maximum time lag in seconds
            regularization (float, int): The regularization paramter (lambda).
        """
        # If the data contains only a single observation, add empty dimension
        if stimulus.ndim == 2 and response.ndim == 2:
            stimulus = np.expand_dims(stimulus, axis=0)
            response = np.expand_dims(response, axis=0)
        delta = 1 / fs
        self.fs = fs
        self.regularization = regularization
        cov_xx = 0
        cov_xy = 0
        if self.direction == 1:
            xs, ys = stimulus, response
        if self.direction == -1:
            xs, ys = response, stimulus
            tmin, tmax = -1 * tmax, -1 * tmin
        for i_trial in range(stimulus.shape[0]):
            x, y = xs[i_trial], ys[i_trial]
            assert x.ndim == 2 and y.ndim == 2
            lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
            # sum covariances matrices across observations
            cov_xx_trial, cov_xy_trial = covariance_matrices(
                x, y, lags, self.zeropad, self.bias
            )
            cov_xx += cov_xx_trial
            cov_xy += cov_xy_trial
        cov_xx /= stimulus.shape[0]
        cov_xy /= stimulus.shape[0]
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
        feature=None,
        average_trials=True,
        average_features=True,
    ):
        """
        Use the trained model to predict the response from the stimulus
        (or vice versa) and optionally estimate the prediction's accuracy.
        Arguments:
            stimulus (np.ndarray | None): stimulus matrix of shape
                trials x samples x features. The first dimension can be
                omitted if there is only a single trial. When using a forward
                model, this must be specified. When using a backward model
                it can be provided to estimate the prediction's error and
                correlation with the actual response.
            response (np.ndarray | None):  response matrix of shape
                trials x samples x features. The first dimension can be omitted
                if there is only a single trial. When using a backward model,
                this must be specified. When using a forward model it can be
                provided to estimate the prediction's error and correlation
                with the actual response.
            lag (int | list of int | None): If not None (default), only use the
                specified lags for prediction. The provided integers are used
                for indexing the elements in self.times.
            feature (int | list of int | None): If not None (default), only use
                the specified features of the stimulus or response for
                prediction. The provided integeres are used to index the
                inputs in the first dimension of self.weights.
            average_trials (bool): If True (default), average correlation
                and error across all trials.
            average_features (bool): If True (default), average correlation
                and error across all prediction features (e.g. channels in
                the case of forward modelling).
        Returns:
            prediction (np.ndarray): Predicted output. Has the same shape as
                the input size of the last dimension (i.e. features) is equal
                to the last dimension in self.weights.
            correlation (float, np.ndarray): If average_trials and
                average_features are True, this is a scalar. Otherwise it's an
                array with one value per trial and feature.
            error (float, np.ndarray):If average_trials and average_features
                are True, this is a scalar. Otherwise it's an array with one
                value per trial and feature.
        """
        # check that inputs are valid
        if self.weights is None:
            raise ValueError("Can't make predictions with an untrained model!")
        if self.direction == 1 and stimulus is None:
            raise ValueError("Need stimulus to predict with a forward model!")
        elif self.direction == -1 and response is None:
            raise ValueError("Need response to predict with a backward model!")
        # if only a single observation, add an empty dimension
        if stimulus is not None:
            if stimulus.ndim == 2:
                stimulus = np.expand_dims(stimulus, axis=0)
        if response is not None:
            if response.ndim == 2:
                response = np.expand_dims(response, axis=0)
        if stimulus is None:
            stimulus = np.repeat(None, response.shape[0])
        if response is None:
            response = np.repeat(None, stimulus.shape[0])
        # create output arrays:
        if self.direction == 1:
            prediction = np.zeros(stimulus.shape[:2] + (self.weights.shape[-1],))
            correlation = np.zeros((stimulus.shape[0], self.weights.shape[-1]))
            error = np.zeros((stimulus.shape[0], self.weights.shape[-1]))
        elif self.direction == -1:
            prediction = np.zeros(response.shape[:2] + (self.weights.shape[-1],))
            correlation = np.zeros((response.shape[0], self.weights.shape[-1]))
            error = np.zeros((response.shape[0], self.weights.shape[-1]))
        # predict y for each trial:
        for i_trial in range(stimulus.shape[0]):
            if self.direction == 1:
                x, y = stimulus[i_trial], response[i_trial]
            elif self.direction == -1:
                x, y = response[i_trial], stimulus[i_trial]

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
            if feature is not None:
                if not isinstance(feature, Iterable):
                    feature = [feature]
                w = w[feature, :, :]
                x_features = len(feature)
                x = x[:, feature]
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
                correlation[i_trial], error[i_trial] = r, err
            prediction[i_trial] = y_pred
        if prediction.shape[0] == 1:  # remove empty dimension
            prediction = prediction[0]
        if y is not None:
            if average_trials is True:
                correlation, error = correlation.mean(0), error.mean(0)
            if average_features is True:
                correlation, error = correlation.mean(-1), error.mean(-1)
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

    def plot_forward_weights(
        self,
        tmin=None,
        tmax=None,
        channels=None,
        axes=None,
        show=True,
        mode="avg",
        kind="line",
    ):
        """
        Plot the weights of a forward model, indicating how strongly the
        neural response is affected by stimulus features at different time
        lags.
        Arguments:
            tmin (None | float): Start of the time window for plotting in
                seconds. If None (default) this is set to 0.05 seconds
                after beginning of self.times.
            tmax (None | float): End of the time window for plotting in
                seconds. If None (default) this is set to 0.05 before
                the end of self.times.
            channels (None | list | int): If an integer or a list of integers,
                only use those channels. If None (default), use all.
            axes (matplotlib.axes.Axes): Axis to plot to. If None is
                provided (default) generate a new plot.
            show (bool): If True (default), show the plot after drawing.
            mode (str): Mode for combining information across channels.
                Can be 'avg' to use the mean or 'gfp' to use global
                field power (i.e. standard deviation across channels).
            kind (str): Type of plot to draw. If 'line' (default), average
                the weights across all stimulus features, if 'image' draw
                a features-by-times plot where the weights are color-coded.
        Returns:
            fig (matplotlib.figure.Figure): If now axes was provided and
                a new figure is created, it is returned.
        """
        if self.direction == -1:
            raise ValueError("Not possible for decoding models!")
        if axes is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = None  # dont create a new figure
        # select time window
        if tmin is None:
            tmin = self.times[0] + 0.05
        if tmax is None:
            tmax = self.times[-1] - 0.05
        start = np.argmin(np.abs(self.times - tmin))
        stop = np.argmin(np.abs(self.times - tmax))
        weights = self.weights[:, start:stop, :]
        # select channels and average if there are multiple
        if isinstance(channels, int):
            weights = weights[:, :, channels]
        else:
            if isinstance(channels, list):
                weights = weights[:, :, channels]
            else:
                weights = weights
            if mode == "avg":
                weights = weights.sum(axis=-1)
            elif mode == "gfp":
                weights = weights.std(axis=-1)
        if kind == "line":
            ax.plot(self.times[start:stop], weights.mean(axis=0))
        elif kind == "image":
            ax.imshow(
                weights,
                origin="lower",
                aspect="auto",
                extent=[tmin, tmax, 0, weights.shape[0]],
            )
        ax.set(xlabel="Time lag [s]")
        if show is True:
            plt.show()
        if fig is not None:
            return fig

    def plot_topography(self, info, stimulus_feature=None):
        try:
            from mne.viz import plot_topomap
        except ImportError:
            print("Topographical plots require MNE-Python!")

        if stimulus_feature is None:
            weights = self.weights.mean(axis=0)
        else:
            weights = self.weights[stimulus_feature, :, :]
        plot_topomap(weights, info)


# define matrix operations
def truncate(x, tminIdx, tmaxIdx):
    """
    the left and right ranges will both be included
    """
    rowSlice = slice(max(0, tmaxIdx), min(0, tminIdx) + len(x))
    output = x[rowSlice]
    return output


def covariance_matrices(x, y, lags, zeropad=True, bias=True):
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    x_lag = lag_matrix(x, lags, zeropad, bias)
    cov_xx = x_lag.T @ x_lag
    cov_xy = x_lag.T @ y
    return cov_xx, cov_xy


def lag_matrix(x, lags, zeropad=True, bias=True):
    """
    Construct a matrix with time lagged input features.
    See also 'lagGen' in mTRF-Toolbox github.com/mickcrosse/mTRF-Toolbox.
    Arguments:
        x (np.ndarray): Input data matrix of shape time x features
        lags (list): Time lags in samples
    lags: a list (or list like supporting len() method) of integers,
         each of them should indicate the time lag in samples.
    zeropad (bool): If True (default) apply zero paddinf to the colums
        with non-zero time lags to ensure causality. Otherwise,
        truncate the matrix.
    bias (bool): If True (default), concatenate an array of ones to
        the left of the array to include a constant bias term in the
        regression.
    Returns:
        lag_matrix (np.ndarray): Matrix of time lagged inputs with shape
            times x number of lags * number of features (+1 if bias==True).
            If zeropad is False, the first dimension is truncated.
    """
    n_lags = len(lags)
    n_samples, n_variables = x.shape
    if max(lags) > n_samples:
        raise ValueError("The maximum lag can't be longer than the signal!")
    lag_matrix = np.zeros((n_samples, n_variables * n_lags))

    for idx, lag in enumerate(lags):
        col_slice = slice(idx * n_variables, (idx + 1) * n_variables)
        if lag < 0:
            lag_matrix[0 : n_samples + lag, col_slice] = x[-lag:, :]
        elif lag > 0:
            lag_matrix[lag:n_samples, col_slice] = x[0 : n_samples - lag, :]
        else:
            lag_matrix[:, col_slice] = x

    if zeropad is False:
        lag_matrix = truncate(lag_matrix, lags[0], lags[-1])

    if bias:
        lag_matrix = np.concatenate([np.ones((lag_matrix.shape[0], 1)), lag_matrix], 1)

    return lag_matrix


def regularization_matrix(size, method="ridge"):
    """
    generates a sparse regularization matrix for the specified method.
    see also regmat.m in https://github.com/mickcrosse/mTRF-Toolbox.
    """
    if method == "ridge":
        regmat = np.identity(size)
        regmat[0, 0] = 0
    elif method == "tikhonov":
        regmat = np.identity(size)
        regmat -= 0.5 * (np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1))
        regmat[1, 1] = 0.5
        regmat[size - 1, size - 1] = 0.5
        regmat[0, 0] = 0
        regmat[0, 1] = 0
        regmat[1, 0] = 0
    else:
        regmat = np.zeros((size, size))
    return regmat
