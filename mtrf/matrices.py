import numpy as np


def _check_data(stimulus=None, response=None, min_len=1, crop=False):
    """
    Check wheter data (stimulus or response) are formatted correctly.
    Parameters
    ----------
        data: list or numpy.ndarray
            Either a two-dimensional samples-by-features array
            or a list of such arrays. If the arrays are only one-dimensional, it is
            assumed that they contain only one feature and a singleton dimension added.
        assert_list: bool
            If True raise an error if `data` is not a list
        assert_len: bool or int
            If an integer, raise an error if the length of `data` does not
            equal that value.
        unisize: bool
            If True, crop all trials to the length of the shortest one to enforce
            equal size.
    Returns
    -------
        data (list): Data in a list of arrays with added singelton dimension if the arrays
            were 1-dimensional.
    """
    for i, data in enumerate([stimulus, response]):
        if isinstance(data, tuple):
            data = list(data)
        elif isinstance(data, np.ndarray):  # convert array to list
            if data.ndim > 2:
                raise ValueError("Array can't have more that three dimensions!")
            else:
                data = [data]
        if data is not None:
            if not all([isinstance(d, np.ndarray) for d in data]):
                raise ValueError("Trials must be arrays!")
            n_trials = len(data)
            if n_trials < min_len:  # check length
                raise ValueError("Data list is too short!")
            min_n = min([len(d) for d in data])
            for j in range(len(data)):
                if data[j].ndim == 1:
                    data[j] = np.expand_dims(data[j], axis=1)
                if crop is True:  # crop all trials to same number of samples
                    data[j] = data[j][:min_n, :]
        if i == 0:
            stimulus = data
        else:
            response = data
    if (stimulus is not None) and (response is not None):
        if (not len(stimulus) == len(response)) or (
            not all([s.shape[0] == r.shape[0] for s, r in zip(stimulus, response)])
        ):
            raise ValueError(
                "Stimulus and response must have the same number of trials and the same number of samples in each trial!"
            )
    return stimulus, response, n_trials


def _get_xy(stimulus, response, tmin=None, tmax=None, direction=1):
    if direction == 1:
        x, y = stimulus, response
    elif direction == -1:
        x, y = response, stimulus
        if tmin is not None and tmax is not None:
            tmin, tmax = -1 * tmax, -1 * tmin
    else:
        raise ValueError("Direction must be 1 or -1.")

    if tmin is not None and tmax is not None:
        return x, y, tmin, tmax
    else:
        return x, y


def truncate(x, min_idx, max_idx):
    """
    Truncate matrix.

    Input matrix is truncated by rows (i.e. the time dimension in a TRF).

    Parameters
    ----------
    x: numpy.ndarray
        Matrix to truncate.
    min_idx: int
        Smallest (time) index to include.
    max_idx: int
        Smallest (time) index to include.

    Returns
    -------
        x_truncated: numpy.ndarray
            Truncated version of ``x``.
    """
    rowSlice = slice(max(0, max_idx), min(0, min_idx) + len(x))
    x_truncated = x[rowSlice]
    return x_truncated


def covariance_matrices(x, y, lags, zeropad=True, preload=True):
    """
    Compute (auto-)covariance of x and y.

    Compute the autocovariance of the time-lagged input x and the covariance of
    x and the output y. When passed a list of trials for x, and y, covariance
    matrices will be computed for each trial.

    Parameters
    ----------
    x: numpy.ndarray or list
        Input data in samples-by-features array or list of such arrays.
    y: numpy.ndarray or list
        Output data in samples-by-features array or list of such arrays.
    lags: list or numpy.ndarray
        Time lags in samples.
    zeropad: bool
        If True (default), pad the input with zeros, if false, truncate the output.

    Returns
    -------
    cov_xx: numpy.ndarray
        Three dimensional autocovariance matrix. 1st dimension's size is the number
        of trials, 2nd and 3rd dimensions' size is lags times features in x.
        If x contains only one trial, the first dimension is empty and will be removed.
    cov_xy: numpy.ndarray
        Three dimensional x-y-covariance matrix. 1st dimension's size is the number
        of trials, 2nd dimension's size is lags times features in x and 3rd dimension's
        size is features in y. If y contains only one trial, the first dimension is
        empty and will be removed.
    """
    x, y, _ = _check_data(x, y)
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    cov_xx, cov_xy = 0, 0
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        x_lag = lag_matrix(x_i, lags, zeropad)
        if preload is True:
            if i == 0:
                cov_xx = np.zeros((len(x), x_lag.shape[-1], x_lag.shape[-1]))
                cov_xy = np.zeros((len(y), x_lag.shape[-1], y_i.shape[-1]))
            cov_xx[i] = x_lag.T @ x_lag
            cov_xy[i] = x_lag.T @ y_i
        else:
            cov_xx += x_lag.T @ x_lag
            cov_xy += x_lag.T @ y_i
    if preload is False:
        cov_xx, cov_xy = cov_xx / len(x), cov_xy / len(x)

    return cov_xx, cov_xy


def lag_matrix(x, lags, zeropad=True, bias=True):
    """
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
    """
    n_lags = len(lags)
    n_samples, n_variables = x.shape
    if max(lags) > n_samples:
        raise ValueError("The maximum lag can't be longer than the signal!")
    lag_matrix = np.zeros((n_samples, n_variables * n_lags))

    for idx, lag in enumerate(lags):
        col_slice = slice(idx * n_variables, (idx + 1) * n_variables)
        if lag < 0:
            lag_matrix[0 : n_samples + lag, col_slice] = x[-lag:, :]
        elif lag > 0:
            lag_matrix[lag:n_samples, col_slice] = x[0 : n_samples - lag, :]
        else:
            lag_matrix[:, col_slice] = x

    if zeropad is False:
        lag_matrix = truncate(lag_matrix, lags[0], lags[-1])

    if bias is not False:
        lag_matrix = np.concatenate([np.ones((lag_matrix.shape[0], 1)), lag_matrix], 1)

    return lag_matrix


def regularization_matrix(size, method="ridge"):
    """
    Generates a sparse regularization matrix for the specified method.

    Parameters
    ----------
    size: int
        Size of the regularization matrix.
    method: str
        Regularization method. Can be 'ridge', 'banded' or 'tikhonov'.

    Returns
    -------
    regmat: numpy.ndarray
        regularization matrix for specified ``size`` and ``method``.
    """
    if method in ["ridge", "banded"]:
        regmat = np.identity(size)
        regmat[0, 0] = 0
    elif method == "tikhonov":
        regmat = np.identity(size)
        regmat -= 0.5 * (np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1))
        regmat[1, 1] = 0.5
        regmat[size - 1, size - 1] = 0.5
        regmat[0, 0] = 0
        regmat[0, 1] = 0
        regmat[1, 0] = 0
    else:
        regmat = np.zeros((size, size))
    return regmat


def banded_regularization(n_lags, coefficients, bands):
    """
    Create regularization matrix for banded ridge regression.

    Parameters
    ----------
    n_lags: int
        Number of time lags
    coefficients: list
        Regularization coefficient for each band. Must be of same length as `bands`.
    bands: list
        Size of the feature bands for which a regularization parameter is fitted, in
        the order they appear in the stimulus matrix. For example, when the stimulus
        is an envelope vector and a 16-band spectrogram, `bands` would be [1, 16].
    """

    if bands is None:
        raise ValueError("Must provide band sizes when using banded ridge regression!")
    if not len(bands) == len(coefficients):
        raise ValueError("Coefficients and bands must be of same size!")
    lag_coefs = []
    # repeat the coefficient for each occurence of the corresponding band
    for c, f in zip(coefficients, bands):
        lag_coefs.append(np.repeat(c, f))
    lag_coefs = np.concatenate(lag_coefs)
    # repeat that sequence for each lag
    diagonal = np.concatenate([lag_coefs for i in range(n_lags)])
    diagonal = np.concatenate([[0], diagonal])
    return np.diag(diagonal)
