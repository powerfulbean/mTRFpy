import random
import time
import sys
import numpy as np
from mtrf.matrices import regularization_matrix, covariance_matrices, _check_data


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
    average=True,
    verbose=True,
):
    """
    Test model accuracy using k-fold cross-validation.

    Input data is randomly shuffled and separated into k parts of with approximately
    the same number of trials. The first k-1 parts are used for training and the kth
    part for testing the model.

    Parameters
    ----------
    model: model.TRF
        Base model used for cross-validation.
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
    regularization: float or int
        Value for the lambda parameter regularizing the regression.
    k: int
        Number of data splits, if -1, do leave-one-out cross-validation.
    seed: int
        Seed for the random number generator.
    average: bool
        If True (default), average correlation and mean squared error across all
        predictions (e.g. channels in the case of forward modelling).

    Returns
    -------
    correlation: float or numpy.ndarray
        When the actual output is provided, correlation is computed per trial or
        averaged, depending on the `average` parameter.
    error: float or numpy.ndarray
        When the actual output is provided, mean squared error is computed per
        trial or averaged, depending on the `average` parameter.
    """
    if seed is not None:
        random.seed(seed)
    stimulus, response = _check_data(stimulus), _check_data(response)
    if model.direction == 1:
        x, y = stimulus, response
    elif model.direction == -1:
        x, y = response, stimulus
        tmin, tmax = -1 * tmax, -1 * tmin
    lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
    cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad, model.bias)
    r, mse = _cross_validate(
        model, x, y, cov_xx, cov_xy, lags, fs, regularization, k, average, verbose
    )
    return r, mse


def _cross_validate(
    model, x, y, cov_xx, cov_xy, lags, fs, regularization, k, average=True, verbose=True
):
    delta = 1 / fs
    regmat = regularization_matrix(cov_xx.shape[-1], model.method)
    regmat *= regularization / delta
    n_trials = len(x)
    k = _check_k(k, n_trials)
    splits = np.arange(n_trials)
    random.shuffle(splits)
    splits = np.array_split(splits, k)
    if average is True:
        r, mse = np.zeros(k), np.zeros(k)
    else:
        r, mse = np.zeros((k, y[0].shape[-1])), np.zeros((k, y[0].shape[-1]))
    for isplit in _progressbar(range(len(splits)), "Cross-validating", verbose=verbose):
        idx_test = splits[isplit]
        idx_train = np.concatenate(splits[:isplit] + splits[isplit + 1 :])  # flatten
        # compute the model for the training set
        cov_xx_hat = cov_xx[idx_train].mean(axis=0)
        cov_xy_hat = cov_xy[idx_train].mean(axis=0)
        weight_matrix = (
            np.matmul(np.linalg.inv(cov_xx_hat + regmat), cov_xy_hat) / delta
        )
        trf = model.copy()
        trf.times, trf.bias, trf.fs = np.array(lags) / fs, weight_matrix[0:1], fs
        trf.weights = weight_matrix[1:].reshape(
            (x[0].shape[-1], len(lags), y[0].shape[-1]), order="F"
        )
        # use the model to predict the test data
        x_test, y_test = [x[i] for i in idx_test], [y[i] for i in idx_test]
        _, r_test, mse_test = trf.predict(x_test, y_test, average=average)
        r[isplit], mse[isplit] = r_test, mse_test
    return r.mean(axis=0), mse.mean(axis=0)


def permutation_distribution(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    n_permute=100,
    k=-1,
    average=True,
    seed=None,
    verbose=True,
):
    if seed:
        np.random.seed(seed)
    stimulus, response = _check_data(stimulus), _check_data(response)
    if model.direction == 1:
        x, y = stimulus, response
    elif model.direction == -1:
        y, x = response, stimulus
    n_trials = len(x)
    k = _check_k(k, n_trials)
    idx = np.arange(n_trials)
    combinations = np.transpose(np.meshgrid(idx, idx)).reshape(-1, 2)
    # only keep the mismatching pairs
    combinations = combinations[~(combinations[:, 0] == combinations[:, 1])]

    for itrial in _progressbar(range(n_trials), "Preparing models", verbose=verbose):
        s, r = stimulus[itrial], response[itrial]
        trf = model.copy()
        trf.train(s, r, fs, tmin, tmax, regularization)
        models.append(trf)

    lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
    for i_x in range(len(x)):
        x_lag = lag_matrix(x[i_x], lags, model.zeropad, model.bias)
        if i_x == 0:
            cov_xx = np.zeros((len(x), x_lag.shape[-1], x_lag.shape[-1]))
            cov_xy = np.zeros((len(combinations), x_lag.shape[-1], y[0].shape[-1]))
        cov_xx[i_x] = x_lag.T @ x_lag
    delta = 1 / fs
    regmat = (
        regularization_matrix(cov_xx.shape[1], model.method) * regularization / delta
    )
    for i, (i_x, i_y) in enumerate(combinations):
        # ensure that x and y have same number of samples
        if x[i_x].shape[0] > y[i_y].shape[0]:
            x[i_x] = x[i_x][: len(y[i_y])]
        elif x[i_x].shape[0] < y[i_y].shape[0]:
            y[i_y] = y[i_y][: len(x[i_x])]
        x_lag = lag_matrix(x[i_x], lags, model.zeropad, model.bias)
        cov_xy[i] = x_lag.T @ y[i_y]
    del x_lag
    if average is True:
        correlations, errors = np.zeros(n_permute), np.zeros(n_permute)
    elif average is False:  # return correlation for each channel/feature
        correlations = np.zeros((n_permute, y[0].shape[-1]))
        errors = np.zeros((n_permute, y[0].shape[-1]))
    for iperm in _progressbar(range(n_permute), "Permuting", verbose=verbose):
        # sample from the covariance matrices
        idx = np.random.choice(len(combinations), len(x), replace=True)
        splits = np.array_split(idx, k)
        perm_corr, perm_err = np.zeros(k), np.zeros(k)
        for isplit in range(k):  # cross-validation
            idx_test = splits[isplit]
            idx_train = np.concatenate(splits[:isplit] + splits[isplit + 1 :])
            # get the sample covariance matrices and average them
            perm_cov_xy = cov_xy[idx_train].mean(axis=0)
            perm_cov_xx = cov_xx[combinations[idx_train][:, 0]].mean(axis=0)
            weight_matrix = (  # compute the regression weights
                np.matmul(np.linalg.inv(perm_cov_xx + regmat), perm_cov_xy) / delta
            )
            x_test = x[combinations[idx_test][0][0]]
            y_test = y[combinations[idx_test][0][0]]
            x_lag = lag_matrix(x_test, lags, model.zeropad, model.bias)
            y_pred = x_lag @ weight_matrix
            if model.zeropad is False:
                y_test = truncate(y_test, lags[0], lags[-1])
            err = np.mean((y_test - y_pred) ** 2, axis=0)
            r = np.mean((y_test - y_test.mean(0)) * (y_pred - y_pred.mean(0)), 0) / (
                y_test.std(0) * y_pred.std(0)
            )
            if average is True:
                err, r = err.mean(), r.mean()
            correlations[iperm] = r
            errors[iperm] = err

    return correlations, erros


def _progressbar(it, prefix="", size=50, out=sys.stdout, verbose=True):
    count = len(it)

    def show(j, verbose):
        x = int(size * j / count)
        if verbose:
            print(
                "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
                end="\r",
                file=out,
                flush=True,
            )

    show(0, verbose)
    for i, item in enumerate(it):
        yield item
        show(i + 1, verbose)
    if verbose:
        print("\n", flush=True, file=out)


def _check_k(k, n_trials):
    if not n_trials > 1:
        raise ValueError("Cross validation requires multiple trials!")
    if n_trials < k:
        raise ValueError("Number of splits can't be greater than number of trials!")
    if k == -1:  # do leave-one-out cross-validation
        k = n_trials
    return k
