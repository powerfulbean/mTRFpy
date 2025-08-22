import random
import sys
from collections.abc import Iterable
from itertools import product
from typing import List, Union
import array_api_compat
from mtrf.matrices import (
    regularization_matrix,
    covariance_matrices,
    banded_regularization,
    _check_data,
    _check_length,
    _get_xy,
    Array,
    ArrayList,
)


def neg_mse(y: Array, y_pred: Array) -> Array:
    """
    Compute negative mean suqare error (mse) between predicted
    and observed data

    Parameters
    ----------
    y: Array
        samples-by-features matrix of observed data.
    y_pred: Array
        samples-by-features matrix of predicted data.

    Returns
    -------
    neg_mse: Array
        Negative mse (-mse) for each feature in y.
    """
    xp = array_api_compat.array_namespace(y)
    mse = xp.mean((y - y_pred) ** 2, axis=0)
    return -mse


def pearsonr(y: Array, y_pred: Array) -> Array:
    """
    Compute Pearson's correlation coefficient between predicted
    and observed data

    Parameters
    ----------
    y: np.ndarray
        samples-by-features matrix of observed data.
    y_pred: np.ndarray
        samples-by-features matrix of predicted data.

    Returns
    -------
    r: np.ndarray
        Pearsons r for each feature in y.
    """
    xp = array_api_compat.array_namespace(y)
    r = xp.mean((y - y.mean(0)) * (y_pred - y_pred.mean(0)), 0) / (
        y.std(0) * y_pred.std(0)
    )
    return r


def crossval(
    model,
    stimulus: ArrayList,
    response: ArrayList,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: Union[int, float],
    k: int = -1,
    seed: Union[int, None] = None,
    average: Union[bool, List[int]] = True,
    verbose: bool = True,
) -> Union[float, Array]:
    """
    Test model metric using k-fold cross-validation.

    Input data is randomly shuffled and separated into k parts of with approximately
    the same number of trials. The first k-1 parts are used for training and the kth
    part for testing the model.

    Parameters
    ----------
    model: TRF
        Base model used for cross-validation.
    stimulus: list of array-like
        Each element must contain one trial's stimulus in a two-dimensional
        samples-by-features array (second dimension can be omitted if there is
        only a single feature.
    respnse: list of array-like
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
        If `False`, return each predicted feature's metric.

    Returns
    -------
    metric: float or array-like
        Metric as computed by the metric function in the attribute `model.metric`.
    """
    stimulus, xps = _check_data(stimulus)
    response, xpr = _check_data(response)
    n_trials = _check_length(stimulus, response)
    assert n_trials >= k, f"Not enough trials for {k}-fold cross-validation"
    if not xps == xpr:
        raise TypeError("stimulus and response trials must be of the same type!")
    else:
        xp = xps
    if isinstance(regularization, Iterable):
        raise ValueError(
            "Crossval only accepts a single scalar for regularization! "
            "For cross-validation with multiple regularization values use `nested_crossval`!"
        )
    if len(stimulus) < 2:
        raise ValueError("Cross-validation requires at least two trials!")
    trf = model.copy()
    if seed is not None:
        random.seed(seed)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = list(range(int(xp.floor(tmin * fs)), int(xp.ceil(tmax * fs)) + 1))
    cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad, trf.preload)
    metric = _crossval(
        model,
        x,
        y,
        cov_xx,
        cov_xy,
        lags,
        fs,
        regularization,
        k,
        xp,
        average,
        verbose,
    )
    return metric


def nested_crossval(
    model,
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
    Unbiased estimate of model accuracy when fitting the regularization parameter.

    This fuction divides the data into k parts and runs two nested
    cross-validation loops: the outer loop selects k-1 parts to optimize the
    regularization value and the kth part to test the final model's accuracy.
    The inner loop uses cross-validation to determine the best regularization
    value as in the `fit` method. The data are rotated so that each of the
    k segments is used to test the final model's accuracy once. The average
    correlation and mean squared error across all folds is an unbiased estimate
    of the model's accuracy because the test data was not part of the optimization
    process.

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
        Number of data splits for cross validation, defaults to 5.
        If -1, do leave-one-out cross-validation.
    average: bool or list or numpy.ndarray
        If True (default), average correlation and mean squared error across all
        predictions (e.g. channels in the case of forward modelling). If `average`
        is an array of integers only average the predicted features at those indices.
    seed: int
        Seed for the random number generator.
    verbose: bool
        If True (default), show a progress bar during fitting.

    Returns
    -------
    metric_test: numpy.ndarray
        Metric as computed by the metric function defined in the attribute
        `TRF.metric` for all k test sets.
    best_regularization: numpy.ndarray
        Optimal regularization values for all k training sets.
    """
    stimulus, xps = _check_data(stimulus)
    response, xpr = _check_data(response)
    n_trials = _check_length(stimulus, response)
    assert (
        n_trials >= k + 1 and n_trials >= 3
    ), f"Not enough trials for {k}-fold nested cross-validation!"
    if not xps == xpr:
        raise TypeError("stimulus and response trials must be of the same type!")
    else:
        xp = xps
    if average is False and not xp.isscalar(regularization):
        raise ValueError("Average must be True or a list of indices!")
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = list(range(int(xp.floor(tmin * fs)), int(xp.ceil(tmax * fs)) + 1))
    if model.method == "banded":
        coefficients = list(product(regularization, repeat=2))
        regularization = [
            banded_regularization(len(lags), c, bands, xp) for c in coefficients
        ]

    if model.preload:
        cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad)
    else:
        cov_xx, cov_xy = None, None

    splits = xp.array_split(xp.arange(n_trials), k)
    n_splits = len(splits)
    metric_test = xp.zeros(n_splits)
    best_regularization = []
    for split_i in range(n_splits):
        idx_test = splits[split_i]
        idx_train_val = xp.concatenate(splits[:split_i] + splits[split_i + 1 :])
        if not xp.isscalar(regularization):
            metric = xp.zeros(len(regularization))
            for ir in _progressbar(
                range(len(regularization)),
                "Hyperparameter optimization",
                verbose=verbose,
            ):
                if cov_xx is not None:
                    cov_xx_train = cov_xx[idx_train_val, :, :]
                    cov_xy_train = cov_xy[idx_train_val, :, :]
                else:
                    cov_xx_train, cov_xy_train = None, None
                metric[ir] = _crossval(
                    model.copy(),
                    [x[i] for i in idx_train_val],
                    [y[i] for i in idx_train_val],
                    cov_xx_train,
                    cov_xy_train,
                    lags,
                    fs,
                    regularization[ir],
                    k - 1,
                    xp,
                    seed=seed,
                    average=average,
                    verbose=verbose,
                )
            regularization_split_i = list(regularization)[xp.argmax(metric)]
        else:
            regularization_split_i = regularization
        model._train(
            [x[i] for i in idx_train_val],
            [y[i] for i in idx_train_val],
            fs,
            tmin,
            tmax,
            regularization_split_i,
        )
        _, metric_test[split_i] = model.predict(
            [stimulus[i] for i in idx_test], [response[i] for i in idx_test]
        )
        best_regularization.append(regularization_split_i)
    return metric_test, best_regularization


def _crossval(
    model,
    x,
    y,
    cov_xx,
    cov_xy,
    lags,
    fs,
    regularization,
    k,
    xp,
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
    splits = xp.arange(n_trials)
    random.shuffle(splits)
    splits = xp.array_split(splits, k)

    if average is True:
        metric = xp.zeros(k)
    else:
        metric = xp.zeros((k, y[0].shape[-1]))

    for isplit in _progressbar(range(len(splits)), "Cross-validating", verbose=verbose):
        idx_val = splits[isplit]
        idx_train = xp.concatenate(splits[:isplit] + splits[isplit + 1 :])  # flatten
        if cov_xx is None:
            x_train = [x[i] for i in idx_train]
            y_train = [y[i] for i in idx_train]
            cov_xx_hat, cov_xy_hat = covariance_matrices(
                x_train, y_train, lags, model.zeropad, preload=False
            )
        else:
            cov_xx_hat = cov_xx[idx_train].mean(axis=0)
            cov_xy_hat = cov_xy[idx_train].mean(axis=0)
        w = xp.matmul(xp.linalg.inv(cov_xx_hat + regmat), cov_xy_hat) / (1 / fs)
        trf = model.copy()
        trf.times, trf.bias, trf.fs = xp.array(lags) / fs, w[0:1], fs
        if trf.bias.ndim == 1:
            trf.bias = xp.expand_dims(trf.bias, 1)
        trf.weights = w[1:].reshape(
            (x[0].shape[-1], len(lags), y[0].shape[-1]), order="F"
        )
        x_test, y_test = [x[i] for i in idx_val], [y[i] for i in idx_val]
        # because we are working with covariance matrices, we have to check direction
        # to pass the right variable as stimulus and response to TRF.predict
        if model.direction == 1:
            _, metric_test = trf.predict(x_test, y_test, None, average)
        else:
            _, metric_test = trf.predict(y_test, x_test, None, average)
        metric[isplit] = metric_test
    return metric.mean(axis=0)


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
        If True (default), average metric across all predicted features (e.g. channels
        in the case of forward modelling). If `average` is an array of indices only
        average the metric for those features. If `False`, return each feature's metric.
    Returns
    -------
    metric: float or numpy.ndarray
        Metric as computed by the metric function in  the attribute `model.metric`
        for each permutation.
    """
    stimulus, xps = _check_data(stimulus)
    response, xpr = _check_data(response)
    n_trials = _check_length(stimulus, response)
    assert n_trials >= k, f"Not enough trials for {k}-fold cross-validation"
    if not xps == xpr:
        raise TypeError("stimulus and response trials must be of the same type!")
    else:
        xp = xps
    if seed:
        xp.random.seed(seed)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    min_len = min([len(x_i) for x_i in x])
    for i, x_i in enumerate(x):
        x[i], y[i] = x_i[:min_len], y[i][:min_len]
    k = _check_k(k, n_trials)
    idx = xp.arange(n_trials)
    combinations = xp.transpose(xp.meshgrid(idx, idx)).reshape(-1, 2)
    models = []
    for c in _progressbar(combinations, "Preparing models", verbose=verbose):
        trf = model.copy()
        trf.train(stimulus[c[0]], response[c[1]], fs, tmin, tmax, regularization)
        models.append(trf)
    metric = xp.zeros(n_permute)
    for iperm in _progressbar(range(n_permute), "Permuting", verbose=verbose):
        idx = []
        for i in range(len(x)):  # make sure each x only appears once
            idx.append(random.choice(xp.where(combinations[:, 0] == i)[0]))
        random.shuffle(idx)
        idx = xp.array_split(idx, k)
        perm_metric = []
        for isplit in range(k):
            idx_val = idx[isplit]
            idx_train = xp.concatenate(idx[:isplit] + idx[isplit + 1 :])
            perm_model = xp.mean([models[i] for i in idx_train])
            stimulus_val = [stimulus[combinations[i][0]] for i in idx_val]
            response_val = [response[combinations[i][1]] for i in idx_val]
            _, fold_metric = perm_model.predict(
                stimulus_val, response_val, None, average
            )
            perm_metric.append(fold_metric)
        metric[iperm] = xp.mean(perm_metric)

    return metric


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
