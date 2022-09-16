import numpy as np


def make_splits(n, k):
    """Split the data into k (almost) equal sets
    Arguments:
        n (int): Number of trials.
        k (int): Number of splits. If k==-1, make one split per trial.
    Returns:
        splits (list): List of arrays where each array contains indices
            of the trials belonging to one split.
    """
    observations = np.arange(n)
    np.random.shuffle(observations)  # randomize trial indices
    if k == -1:  # do leave-one-out cross validation
        splits = [np.array(obs) for obs in observations]
    else:
        splits = np.array_split(observations, k)
    return splits


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
    splits = make_splits(stimulus.shape[0], k)
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
