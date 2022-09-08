import numpy as np


# define matrix operations
def truncate(x, tminIdx, tmaxIdx):
    """
    the left and right ranges will both be included
    """
    rowSlice = slice(max(0, tmaxIdx), min(0, tminIdx) + len(x))
    output = x[rowSlice]
    return output


def covariance_matrices(x, y, lags, zeropad=True, bias=True):
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    x_lag = lag_matrix(x, lags, zeropad, bias)
    cov_xx = x_lag.T @ x_lag
    cov_xy = x_lag.T @ y
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

    if bias:
        lag_matrix = np.concatenate([np.ones((lag_matrix.shape[0], 1)), lag_matrix], 1)

    return lag_matrix


def regularization_matrix(size, method="ridge"):
    """
    generates a sparse regularization matrix for the specified method.
    see also regmat.m in https://github.com/mickcrosse/mTRF-Toolbox.
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


def banded_regularization_coefficients(n_lags, coefficients, features, bias):
    """
    Create a diagonal matrix with the regularization coefficients for banded ridge regression.
    Arguments:
        n_lags (int): Number of time lags
        coefficients (list): regularization coefficient for each feature type.
            Must be of same length as `bands`.
        features (list): Size of feature types in the order that they appear in the
            stimulus matrix (e.g. if the stimuli would consist of a 16 band spectrogram
            and an onset vector, features would be [16, 1]).
        bias (bool): Wether the TRF model includes a bias term.
    """
    if not len(features) == len(coefficients):
        raise ValueError("Coefficients and features must be of same size!")
    lag_coefs = []
    # repeat the coefficient for each occurence of the corresponding feature
    for c, f in zip(coefficients, features):
        lag_coefs.append(np.repeat(c, f))
    lag_coefs = np.concatenate(lag_coefs)
    # repeat that sequence for each lag
    diagonal = np.concatenate([lag_coefs for i in range(n_lags)])
    if bias:
        diagonal = np.concatenate([[0], diagonal])
    return np.diag(diagonal)
