import random
import time
import sys
import numpy as np
from mtrf.matrices import _check_data, lag_matrix


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
    ntrials = len(response)
    if not ntrials > 1:
        raise ValueError("Cross validation requires multiple trials!")
    if ntrials < k:
        raise ValueError("Number of splits can't be greater than number of trials!")
    if ntrials == k:  # do leave-one-out cross-validation
        k = -1
    models = []  # compute the TRF for each trial
    if verbose:
        print("\n")
    for itrial in _progressbar(range(ntrials), "Preparing models", verbose=verbose):
        s, r = stimulus[itrial], response[itrial]
        trf = model.copy()
        trf.train(s, r, fs, tmin, tmax, regularization)
        models.append(trf)
    splits = list(range(ntrials))
    random.shuffle(splits)
    if k != -1:
        splits = np.array_split(splits, k)
        splits = [list(s) for s in splits]
    else:
        splits = [[s] for s in splits]
    correlations, errors = [], []
    for isplit in _progressbar(range(len(splits)), "Cross-validating", verbose=verbose):
        idx_test = splits[isplit]
        idx_train = splits[:isplit] + splits[isplit + 1 :]
        if all(isinstance(x, list) for x in idx_train):  # flatten list of lists
            idx_train = [idx for split in idx_train for idx in split]

        trf = sum([models[i] for i in idx_train]) / len(idx_train)
        _, correlation, error = trf.predict(
            [stimulus[i] for i in idx_test],
            [response[i] for i in idx_test],
            average=average,
        )
        correlations.append(correlation)
        errors.append(error)
    return np.mean(correlations, axis=0), np.mean(errors, axis=0)


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

    idx = np.arange(len(stimulus))
    combinations = np.transpose(np.meshgrid(idx, idx)).reshape(-1, 2)
    # only keep the mismatching pairs
    combinations = combinations[~(combinations[:, 0] == combinations[:, 1])]
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
    for iperm in _progressbar(range(n_permute), "Permuting", verbose=verbose):
        # sample from the covariance matrices
        idx = [
            np.random.choice(np.where(combinations[:, 0] == i_x)[0])
            for i_x in range(len(x))
        ]
        idx = np.random.choice(len(combinations), len(x), replace=True)
        x_idx = combinations[idx][:, 0]
        perm_cov_xy = cov_xy[idx].mean(axis=0)
        perm_cov_xx = cov_xx[combinations[idx][:, 0]].mean(axis=0)
        weight_matrix = (
            np.matmul(np.linalg.inv(perm_cov_xx + regmat), perm_cov_xy) / delta
        )

        self.bias = weight_matrix[0:1]
        self.weights = weight_matrix[1:].reshape(
            (x.shape[1], len(lags), y.shape[1]), order="F"
        )
    return correlations, erros


def _progressbar(it, prefix="", size=50, out=sys.stdout, verbose=True):  # Python3.3+
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
