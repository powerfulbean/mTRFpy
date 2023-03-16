import random
import time
import sys
import numpy as np
from mtrf.matrices import _check_data


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
    print("\n")
    for itrial in _progressbar(range(ntrials), "Preparing models", size=61):
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
    print("\n")
    for isplit in _progressbar(range(len(splits)), "Cross-validating", size=61):
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


def _progressbar(it, prefix="", size=50, out=sys.stdout):  # Python3.3+
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)
