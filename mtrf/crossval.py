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
    if not (isinstance(stimulus, list) and isinstance(response, list)):
        raise ValueError(
            "Cross validation requires a list of multiple trials for stimulus and response!"
        )
    if seed is not None:
        random.seed(seed)
    stimulus, response = _check_data(stimulus), _check_data(response)
    ntrials = len(response)

    splits = list(range(ntrials))
    random.shuffle(splits)
    if k != -1:
        splits = np.array_split(splits, k)
        splits = [list(s) for s in splits]
    correlations, errors = [], []
    print("\n")
    for isplit in _progressbar(range(len(splits)), "Cross-validation", size=61):
        split = splits[isplit]
        idx_test = list(split)
        idx_train = splits[:isplit] + splits[isplit + 1 :]
        if all(isinstance(x, list) for x in idx_train):
            idx_train = [idx for split in idx_train for idx in split]

        trf = model.copy()
        trf.train(
            [stimulus[i] for i in idx_train],
            [response[i] for i in idx_train],
            fs,
            tmin,
            tmax,
            regularization,
        )
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
