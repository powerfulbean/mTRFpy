import random
import sys
import numpy as np
from mtrf.matrices import (
    regularization_matrix,
    covariance_matrices,
    _check_data,
    _get_xy,
)


def mean_squared_error(y, y_pred):
    mse = np.mean((y - y_pred) ** 2, axis=0)
    return mse


def one_minus_r(y, y_pred):
    """
    Compute one minus the Pearson's correlation coefficient between predicted
    and observed data

    Parameters
    ----------
    y: np.ndarray
        samples-by-features matrix of observed data.
    y_pred: np.ndarray
        samples-by-features matrix of predicted data.

    Returns
    -------
    inv_r: np.ndarray
        Inverse Pearsons r (1-r) for each feature in y.
    """
    r = np.mean((y - y.mean(0)) * (y_pred - y_pred.mean(0)), 0) / (
        y.std(0) * y_pred.std(0)
    )
    return 1 - r


def cross_validate(
    model,
    stimulus,
    response,
    fs=None,
    tmin=None,
    tmax=None,
    regularization=None,
    k=-1,
    seed=None,
    average=True,
    verbose=True,
):
    """
    Test model loss using k-fold cross-validation.

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
    average: bool or list or numpy.ndarray
        If True (default), average correlation and mean squared error across all
        predictions (e.g. channels in the case of forward modelling). If `average`
        is an array of integers only average the predicted features at those indices.
        If `False`, return each predicted feature's loss.

    Returns
    -------
    loss: float or numpy.ndarray
        Loss as computed by the loss function in  the attribute `model.loss_function`.
    """
    trf = model.copy()
    trf.bias, trf.weights = None, None
    fs, tmin, tmax, regularization = _check_attr(trf, fs, tmin, tmax, regularization)
    if seed is not None:
        random.seed(seed)
    stimulus, response, _ = _check_data(stimulus, response, min_len=2)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
    cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad, trf.preload)
    loss = _cross_validate(
        model,
        x,
        y,
        cov_xx,
        cov_xy,
        lags,
        fs,
        regularization,
        k,
        average,
        verbose,
    )
    return loss


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

    reg_mat_size = x[0].shape[-1] * len(lags) + 1
    regmat = regularization_matrix(reg_mat_size, model.method)
    regmat *= regularization / (1 / fs)

    n_trials = len(x)
    k = _check_k(k, n_trials)
    splits = np.arange(n_trials)
    random.shuffle(splits)
    splits = np.array_split(splits, k)

    if average is True:
        loss = np.zeros(k)
    else:
        loss = np.zeros((k, y[0].shape[-1]))

    for isplit in _progressbar(range(len(splits)), "Cross-validating", verbose=verbose):
        idx_val = splits[isplit]
        idx_train = np.concatenate(splits[:isplit] + splits[isplit + 1 :])  # flatten
        if cov_xx is None:
            x_train = [x[i] for i in idx_train]
            y_train = [y[i] for i in idx_train]
            cov_xx_hat, cov_xy_hat = covariance_matrices(
                x_train, y_train, lags, model.zeropad, preload=False
            )
        else:
            cov_xx_hat = cov_xx[idx_train].mean(axis=0)
            cov_xy_hat = cov_xy[idx_train].mean(axis=0)
        w = np.matmul(np.linalg.inv(cov_xx_hat + regmat), cov_xy_hat) / (1 / fs)
        trf = model.copy()
        trf.times, trf.bias, trf.fs = np.array(lags) / fs, w[0:1], fs
        if trf.bias.ndim == 1:
            trf.bias = np.expand_dims(trf.bias, 1)
        trf.weights = w[1:].reshape(
            (x[0].shape[-1], len(lags), y[0].shape[-1]), order="F"
        )
        x_test, y_test = [x[i] for i in idx_val], [y[i] for i in idx_val]
        # because we are working with covariance matrices, we have to check direction
        # to pass the right variable as stimulus and response to TRF.predict
        if model.direction == 1:
            _, loss_test = trf.predict(x_test, y_test, None, average)
        elif model.direction == -1:
            _, loss_test = trf.predict(y_test, x_test, None, average)
        loss[isplit] = loss_test
    return loss.mean(axis=0)


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
    seed=None,
    average=True,
    verbose=True,
):
    """
    Estimate the distribution of correlation coefficients and mean squared error
    under random permutation.

    For each permutation, stimulus and response trials are randomly shuffled and
    split into `k` segments. Then `k-1` segments are used to train and the remaining
    segment is used to test the model. The resulting permutation distribution reflects
    the expected correlation and error if there is no causal relationship between
    stimulus and response. To save time, the models are computed for all possible
    combinations of response and stimulus and then sampled and averaged during
    permutation.

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
    average: bool or list or numpy.ndarray
        If True (default), average loss across all predicted features (e.g. channels
        in the case of forward modelling). If `average` is an array of indices only
        average the loss for those features. If `False`, return each feature's loss.
    Returns
    -------
    loss: float or numpy.ndarray
        Loss as computed by the loss function in  the attribute `model.loss_function`
        for each permutation.
    """
    if seed:
        np.random.seed(seed)
    stimulus, response, n_trials = _check_data(stimulus, response, min_len=2, crop=True)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    min_len = min([len(x_i) for x_i in x])
    for i in range(len(x)):
        x[i], y[i] = x[i][:min_len], y[i][:min_len]
    k = _check_k(k, n_trials)
    idx = np.arange(n_trials)
    combinations = np.transpose(np.meshgrid(idx, idx)).reshape(-1, 2)
    models = []
    for c in _progressbar(combinations, "Preparing models", verbose=verbose):
        trf = model.copy()
        trf.train(stimulus[c[0]], response[c[1]], fs, tmin, tmax, regularization)
        models.append(trf)
    loss = np.zeros(n_permute)
    for iperm in _progressbar(range(n_permute), "Permuting", verbose=verbose):
        idx = []
        for i in range(len(x)):  # make sure each x only appears once
            idx.append(random.choice(np.where(combinations[:, 0] == i)[0]))
        random.shuffle(idx)
        idx = np.array_split(idx, k)
        perm_loss = []
        for isplit in range(len(idx)):
            idx_val = idx[isplit]
            idx_train = np.concatenate(idx[:isplit] + idx[isplit + 1 :])
            perm_model = np.mean([models[i] for i in idx_train])
            stimulus_val = [stimulus[combinations[i][0]] for i in idx_val]
            response_val = [response[combinations[i][1]] for i in idx_val]
            _, fold_loss = perm_model.predict(stimulus_val, response_val)
            perm_loss.append(fold_loss)
        loss[iperm] = np.mean(perm_loss)

    return loss


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


def _check_attr(model, fs, tmin, tmax, regularization):
    if fs is None:
        fs = model.fs
    if tmin is None and isinstance(model.times, np.ndarray):
        tmin = np.abs(model.times).min()
    if tmax is None and isinstance(model.times, np.ndarray):
        tmax = np.abs(model.times).max()
    if regularization is None:
        regularization = model.regularization
    if any([x is None for x in [fs, tmin, tmax, regularization]]):
        raise ValueError(
            "Specify parameters `fs`, `tmin`, `tmax` and `regularization` when using and untrained model!"
        )
    return fs, tmin, tmax, regularization
