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
        average (bool): If True (default), average correlation
            and error across all predictions (e.g. channels in
            the case of forward modelling) to get a single score.
    Returns:
        correlations (np.ndarray | float): Correlation between the actual
            and predicted output. If average_features or average_splits
            is False, a separate value for each feature / split is returned
        errors (np.array | float): Mean squared error between the actual
            and predicted output. If average_features or average_splits
            is False, a separate value for each feature / split is returned
    """
    if seed is not None:
        random.seed(seed)
    stimulus, response = _check_data(stimulus), _check_data(response)
    ntrials = len(response)
    if not ntrials > 1:
        raise ValueError("Cross validation requires multiple trials!")
    if ntrials < k:
        raise ValueError("Number of splits can't be greater than number of trials!")
    if k == -1: # do leave-one-out cross-validation
        k = ntrials
    models = []  # compute the TRF for each trial
    print("\n")
    for itrial in _progressbar(range(ntrials), "Preparing models", size=61):
        s, r = stimulus[itrial], response[itrial]
        trf = model.copy()
        trf.train(s, r, fs, tmin, tmax, regularization)
        models.append(trf)
    splits = np.arange(ntrials)
    random.shuffle(splits)
    splits = np.array_split(splits, k)
    correlations, errors = [], []
    print("\n")
    for isplit in _progressbar(range(len(splits)), "Cross-validating", size=61):
        idx_test = splits[isplit]
        # flatten list of lists
        idx_train = np.concatenate(splits[:isplit] + splits[isplit + 1 :])
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
