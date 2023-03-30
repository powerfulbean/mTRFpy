import random
import time
import sys
import numpy as np
from mtrf.matrices import (
    regularization_matrix,
    covariance_matrices,
    lag_matrix,
    _check_data,
    _get_xy,
)


def cross_validate(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    k=-1,
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
    model,
    x,
    y,
    cov_xx,
    cov_xy,
    lags,
    fs,
    regularization,
    k,
    average=True,
    verbose=True,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
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
        idx_val = splits[isplit]
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
        x_test, y_test = [x[i] for i in idx_val], [y[i] for i in idx_val]
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
    n_permute,
    k=-1,
    average=True,
    seed=None,
    verbose=True,
):
    """
    Estimate the distribution of correlation coefficients and mean squared error
    under random permutation.
    """
    if seed:
        np.random.seed(seed)
    stimulus, response = _check_data(stimulus), _check_data(response)
    xs, ys, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, self.direction)
    min_len = min([len(x) for x in xs])
    for i in range(len(xs)):
        xs[i], ys[i] = xs[i][:min_len], ys[i][:min_len]
    n_trials = len(xs)
    k = _check_k(k, n_trials)
    idx = np.arange(n_trials)
    combinations = np.transpose(np.meshgrid(idx, idx)).reshape(-1, 2)
    # combinations = combinations[combinations[:, 0] != combinations[:, 1]]
    lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
    # precompute the predictor autocovaraince for all trials
    for i_x in range(len(xs)):
        x_lag = lag_matrix(xs[i_x], lags, model.zeropad, model.bias)
        if i_x == 0:
            cov_xx = np.zeros((len(xs), x_lag.shape[-1], x_lag.shape[-1]))
        cov_xx[i_x] = x_lag.T @ x_lag
    # precompute the predictor-estimand covariance for all combinations
    for i_c, (i_x, i_y) in enumerate(combinations):
        x_lag = lag_matrix(xs[i_x], lags, model.zeropad, model.bias)
        if i_c == 0:
            cov_xy = np.zeros((len(combinations), x_lag.shape[-1], ys[0].shape[-1]))
        if model.zeropad is False:
            ys[i_y] = truncate(ys[i_y], lags[0], lags[-1])
        cov_xy[i_c] = x_lag.T @ ys[i_y]
    del x_lag
    r, mse = np.zeros(n_permute), np.zeros(n_permute)
    for iperm in _progressbar(range(n_permute), "Permuting", verbose=verbose):
        # sample from all possible permutation matrices
        idx = []
        for i in range(len(xs)):  # make sure eachx x only appears once
            idx.append(random.choice(np.where(combinations[:, 0] == i)[0]))
        random.shuffle(idx)
        # idx = np.random.choice(len(combinations), len(xs), replace=True)
        idx = np.array_split(idx, k)
        perm_cov_xy = cov_xy[
            np.concatenate(idx[1:])
        ]  # the fist split is used for testing
        perm_cov_xx = cov_xx[[combinations[i, 0] for i in np.concatenate(idx[1:])]]
        reg_r, reg_mse = np.zeros(len(regularization)), np.zeros(len(regularization))
        for ireg, reg in enumerate(regularization):
            sample_r, sample_mse = _cross_validate(
                model,
                [xs[combinations[i, 0]] for i in np.concatenate(idx[1:])],
                [ys[combinations[i, 1]] for i in np.concatenate(idx[1:])],
                perm_cov_xx,
                perm_cov_xy,
                lags,
                fs,
                reg,
                k - 1,
                average,
                False,
            )
            reg_r[ireg], reg_mse[ireg] = sample_r, sample_mse
        # estimate accuracy best best regularization value on test set
        best_reg = regularization[np.argmin(reg_mse)]
        regmat = regularization_matrix(cov_xx.shape[-1], model.method) + best_reg
        weight_matrix = np.matmul(
            np.linalg.inv(perm_cov_xx.mean(axis=0) + regmat), perm_cov_xy.mean(axis=0)
        )
        for i in idx[0]:
            x, y = xs[combinations[i, 0]], ys[combinations[i, 1]]
            x_lag = lag_matrix(x, lags, model.zeropad, model.bias)
            y_pred = x_lag @ weight_matrix
            if model.zeropad is False:
                y = truncate(y, lags[0], lags[-1])
            sample_mse = np.mean((y - y_pred) ** 2, axis=0)
            sample_r = np.mean((y - y.mean(0)) * (y_pred - y_pred.mean(0)), 0) / (
                y.std(0) * y_pred.std(0)
            )
            if isinstance(average, list) or isinstance(average, np.ndarray):
                sample_mse, sample_r = sample_mse[average], sample_r[average]
            r[iperm] += sample_r.mean()
            mse[iperm] += sample_mse.mean()
        r[iperm], mse[iperm] = r[iperm] / len(idx), mse[iperm] / len(idx)

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
