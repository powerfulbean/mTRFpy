# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""
import copy
import numpy as np
from matplotlib import pyplot as plt
from . import Tools
try:
    from tqdm import tqdm
except ImportError:
    tdqm = False


def cross_validate(model, stimulus, response, fs, tmin, tmax,
                   regularization, splits=5, test_size=0.1, random_state=None):

    if not stimulus.ndim == 3 and response.ndim == 3:
        raise ValueError('Arrays must be 3D with'
                         'observations x samples x features!')
    if stimulus.shape[0:2] != response.shappe[0:2]:
        raise ValueError('Stimulus and response must have same number of'
                         'samples and observations!')
    observations = np.arange(stimulus.shape[0])
    if splits == -1:  # do leave-one-out cross validation
        idx_test = observations
        idx_train = idx_test[1:] - (idx_test[:, None] >= idx_test[1:])
    else:
        n_test = int(stimulus.shape[0] * test_size)
        n_train = stimulus.shape[0] - n_test
        idx_test = np.zeros((splits, n_test))
        idx_train = np.zeros((splits, n_train))
        for i in range(splits):
            train = np.random.choice(observations, n_train, replace=False)
            test = np.array(list(set(observations) - set(train)))
            idx_test[i], idx_train[i] = test, train

    if tqdm is not False:
        folds = tqdm(range(idx_train.shape[0]))
    else:
        folds = range(idx_train.shape[0])
    models = []
    correlations = np.zeros(idx_train.shape[0])
    errors = np.zeros(idx_train.shape[0])
    for fold in folds:
        trf = model.copy()
        trf.train(stimulus[idx_train], response[idx_train], tmin, tmax,
                  regularization)
        models.append(trf)
        fold_correlation, fold_error = trf.predict(
            stimulus[idx_test], response[idx_test])
        correlations[fold], errors[fold] = fold_correlation, fold_error
    return models, correlations, errors


class TRF:
    '''
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

    '''
    def __init__(self, direction=1, kind='multi', zeropad=True,
                 method='ridge'):
        self.weights = None
        self.bias = None
        self.times = None
        self.fs = -1
        if direction in [1, -1]:
            self.direction = direction
        else:
            raise ValueError('Parameter direction must be either 1 or -1!')
        if kind in ['multi', 'single']:
            self.kind = kind
        else:
            raise ValueError(
                    'Paramter kind must be either "multi" or "single"!')
        if isinstance(zeropad, bool):
            self.zeropad = zeropad
        else:
            raise ValueError('Parameter zeropad must be boolean!')
        if method in ['ridge', 'tikhonov']:
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
            raise TypeError('Can only add to another TRF instance!')
        if not (self.direction == trf.direction) and (self.kind == trf.kind):
            raise ValueError('Added TRFs must be of same kind and direction!')
        trf_new = self.copy()
        trf_new.weights += trf.weights
        trf_new.bias += trf.bias
        return trf_new

    def __truediv__(self, num):
        trf_new = self.copy()
        trf_new.weights /= num
        trf_new.bias /= num
        return trf_new

    def __copy__(self):
        return copy.deepcopy(self)

    def fit(self, stimulus, response, fs, tmin, tmax, regularization,
            splits=5, random_state=42):
        if not stimulus.ndim == 3 and response.ndim == 3:
            raise ValueError('TRF fitting requires 3-dimensional arrays'
                             'for stimulus and response with the shape'
                             'n_stimuli x n_sammples x n_features.')
        if np.isscalar(regularization):
            correlation, error = cross_validate(
                stimulus, response, self.direction, fs, tmin, tmax,
                regularization, splits, random_state)
        else:  # run cross-validation once per regularization parameter
            correlation, error = [], []
            if tqdm is not False:
                regularization = tqdm(regularization, leave=False,
                                      desc='fitting regularization parameter')
            for r in regularization:
                reg_correlation, reg_error = cross_validate(
                        stimulus, response, self.direction, fs, tmin, tmax, r,
                        splits, random_state)
                correlation.append(reg_correlation)
                error.append(reg_error)
            return correlation, error

    def train(self, stimulus, response, fs, tmin, tmax, regularization):
        '''
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
        '''
        if self.direction == 1:
            x, y = stimulus, response
        elif self.direction == -1:
            x, y = response, stimulus
            tmin, tmax = -1 * tmax, -1 * tmin

        lags = list(range(int(np.floor(tmin*fs)), int(np.ceil(tmax*fs)) + 1))
        cov_xx, cov_xy = covariance_matrices(x, y, lags)
        delta = 1/fs
        regmat = regularization_matrix(cov_xx.shape[1], self.method)
        regmat *= regularization / delta
        # calculate reverse correlation:
        weight_matrix = np.matmul(
                np.linalg.inv(cov_xx + regmat), cov_xy) / delta
        self.bias = weight_matrix[0:1]
        self.weights = weight_matrix[1:].reshape(
                (x.shape[1], len(lags), y.shape[1]), order='F')
        self.times = np.array(lags)/fs
        self.fs = fs

    def predict(self, stimulus=None, response=None, lag=None, channel=None):
        if self.weights is None:
            raise ValueError("Can't make predictions with an untrained model!")
        if self.direction == 1:
            if stimulus is None:
                raise ValueError(
                        "Need stimulus to predict with a forward model!")
            else:
                x, y = stimulus, response
        elif self.direction == -1:
            if response is None:
                raise ValueError(
                        "Need response to predict with a backward model!")
            else:
                x, y = stimulus, response

        x_samples, x_features = x.shape
        if y is None:
            y_samples = x_samples
            y_features = self.weights.shape[0]
        else:
            y_samples, y_features = y.shape

        lags = list(range(int(np.floor(self.times[0]*self.fs)),
                          int(np.ceil(self.times[-1]*self.fs)) + 1))
        delta = 1/self.fs

        w = self.weights.copy()
        if lag is not None:  # select lag and corresponding weights
            lags = [lags[lag]]
            w = w[:, lag:lag+1, :]
        if channel is not None:
            w = w[channel:channel+1, :, :]
            x_features = 1
            x = np.expand_dims(x[:, channel], axis=1)
        w = np.concatenate([
            self.bias,
            w.reshape(x_features*len(lags), y_features, order='F')
            ])*delta
        x_lag = lag_matrix(x, lags, self.zeropad)
        y_pred = x_lag @ w
        if y is not None:
            if self.zeropad is False:
                y = truncate(y, lags[0], lags[-1])
            mse = np.sum(np.abs(y - y_pred)**2, 0)/len(y)
            r = (np.mean((y-y.mean(0))*(y_pred-y_pred.mean(0)), 0) /
                 (y.std(0)*y_pred.std(0)))
            return y_pred, r, mse
        else:
            return y_pred

    def save(self, path, name):
        output = dict()
        for i in self.__dict__:
            output[i] = self.__dict__[i]

        Tools.saveObject(output, path, name, '.mtrf')

    def load(self, path):
        temp = Tools.loadObject(path)
        for i in temp:
            setattr(self, i, temp[i])

    def copy(self):
        oTRF = TRF()
        for k, v in self.__dict__.items():
            value = v
            if getattr(v, 'copy', None) is not None:
                value = v.copy()
            setattr(oTRF, k, value)
        return oTRF

    def plot_forward_weights(self, tmin=None, tmax=None, channels=None,
                             axes=None, show=True, mode='avg', kind='line'):
        '''
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
        '''
        if self.direction == -1:
            raise ValueError('Not possible for decoding models!')
        if axes is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = None  # dont create a new figure
        # select time window
        if tmin is None:
            tmin = self.times[0] + 0.05
        if tmax is None:
            tmax = self.times[-1] - 0.05
        start = np.argmin(np.abs(self.times-tmin))
        stop = np.argmin(np.abs(self.times-tmax))
        weights = self.weights[:, start:stop, :]
        # select channels and average if there are multiple
        if isinstance(channels, int):
            weights = weights[:, :, channels]
        else:
            if isinstance(channels, list):
                weights = weights[:, :, channels]
            else:
                weights = weights
            if mode == 'avg':
                weights = weights.sum(axis=-1)
            elif mode == 'gfp':
                weights = weights.std(axis=-1)
        if kind == 'line':
            ax.plot(self.times[start:stop], weights.mean(axis=0))
        elif kind == 'image':
            ax.imshow(weights, origin='lower', aspect='auto',
                      extent=[tmin, tmax, 0, weights.shape[0]])
        ax.set(xlabel='Time lag [s]')
        if show is True:
            plt.show()
        if fig is not None:
            return fig

    def plot_topography(self, info, stimulus_feature=None):
        try:
            from mne.viz import plot_topomap
        except ImportError:
            print('Topographical plots require MNE-Python!')

        if stimulus_feature is None:
            weights = self.weights.mean(axis=0)
        else:
            weights = self.weights[stimulus_feature, :, :]
        plot_topomap(weights, info)


# define matrix operations
def truncate(x, tminIdx, tmaxIdx):
    '''
    the left and right ranges will both be included
    '''
    rowSlice = slice(max(0, tmaxIdx), min(0, tminIdx) + len(x))
    output = x[rowSlice]
    return output


def covariance_matrices(x, y, lags, zeropad=True):
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    x_lag = lag_matrix(x, lags, zeropad)
    cov_xx = x_lag.T @ x_lag
    cov_xy = x_lag.T @ y
    return cov_xx, cov_xy


def lag_matrix(x, lags, zeropad=True, bias=True):
    '''
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
    '''
    n_lags = len(lags)
    n_samples, n_variables = x.shape
    if max(lags) > n_samples:
        raise ValueError("The maximum lag can't be longer than the signal!")
    lag_matrix = np.zeros((n_samples, n_variables*n_lags))

    for idx, lag in enumerate(lags):
        col_slice = slice(idx * n_variables, (idx + 1) * n_variables)
        if lag < 0:
            lag_matrix[0:n_samples + lag, col_slice] = x[-lag:, :]
        elif lag > 0:
            lag_matrix[lag:n_samples, col_slice] = x[0:n_samples-lag, :]
        else:
            lag_matrix[:, col_slice] = x

    if zeropad is False:
        lag_matrix = truncate(lag_matrix, lags[0], lags[-1])

    if bias:
        lag_matrix = np.concatenate(
                [np.ones((lag_matrix.shape[0], 1)), lag_matrix], 1)

    return lag_matrix


def regularization_matrix(size, method='ridge'):
    '''
    generates a sparse regularization matrix for the specified method.
    see also regmat.m in https://github.com/mickcrosse/mTRF-Toolbox.
    '''
    if method == 'ridge':
        regmat = np.identity(size)
        regmat[0, 0] = 0
    elif method == 'tikhonov':
        regmat = np.identity(size)
        regmat -= 0.5 * (np.diag(np.ones(size-1), 1) +
                         np.diag(np.ones(size-1), -1))
        regmat[1, 1] = 0.5
        regmat[size-1, size-1] = 0.5
        regmat[0, 0] = 0
        regmat[0, 1] = 0
        regmat[1, 0] = 0
    else:
        regmat = np.zeros((size, size))
    return regmat
