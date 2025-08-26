from typing import TypeVar, List, Union, Tuple, Literal
from types import ModuleType
import array_api_compat
from array_api_compat import is_array_api_obj

Array = TypeVar("Array")
ArrayList = List[Array]


def _check_data(data: Union[ArrayList, Array]) -> Tuple[ArrayList, ModuleType]:
    """
    Ensure correct data formatting

    Parameters
    ----------
        data: array-like or list of array-like
            Either a two-dimensional samples-by-features array or a list of such arrays.
            If the arrays are one-dimensional a singleton dimension added.
    Returns
    -------
        list of array-like
            Data in a list of arrays with added singelton dimension if the arrays
            were 1-dimensional.
        module
            The namespace for the array implementation of `data`.
    Raises
    ------
    ValueError
        When arrays have more than 2 dimensions or the list of arrays is
        shorter than min_len.
    TypeError
        When trials are not all arrays.

    """
    xp = array_api_compat.array_namespace(*data)
    if is_array_api_obj(data):
        if data.ndim > 2:
            raise ValueError("Array can't have more that three dimensions!")
        else:
            data = [data]
    if not all([is_array_api_obj(d) for d in data]):
        raise TypeError("Trials must be arrays!")
    for i, d in enumerate(data):
        if d.ndim == 1:
            data[i] = xp.expand_dims(d, axis=1)
    return data, xp


def _check_length(stimulus, response, crop=True) -> Tuple[ArrayList, ArrayList, int]:
    """
    Assert stimulus and response have the same number of trials with the same length.

    Parameters
    ----------
        stimulus: list of array-like
            List of stimulus trials.
        response: list of array-like
            List of response trials.
        crop: bool
            If True (default) crop each trial so that stimulus and response
            have the same length.

    Returns
    -------
        list of array-like
            Trial list of stimulus features
        list of array-like
            Trial list of response features
        int
            The number of trials for stimulus and response

    Raises
    ------
        AssertionError
            If stimulus and response don't have the same number of trials
            of the trials don't have the same length.
    """

    assert len(stimulus) == len(
        response
    ), "stimulus and response must have the same number of trials!"
    assert all(
        [s.shape[0] == r.shape[0] for s, r in zip(stimulus, response)]
    ), "stimulus and response trials must have the same length!"
    if crop is True:
        min_n = [min(len(r), len(s)) for r, s in zip(response, stimulus)]
        for i, (r, s, n) in enumerate(zip(response, stimulus, min_n)):
            stimulus[i] = s[i][:n, :]
            response[i] = r[i][:n, :]
    return stimulus, response, len(stimulus)


def _get_xy(
    stimulus: ArrayList,
    response: ArrayList,
    tmin: float,
    tmax: float,
    direction: Literal[1, -1],
) -> Tuple[ArrayList, ArrayList, float, float]:
    """
    Assign `stimulus` and `response` to `x` and `y` depending on the `direction`.

    Parameters
    ----------
        stimulus: list of array-like
            List of stimulus trials.
        response: list of array-like
            List of response trials.
        tmin: float
            Minimum time lag in seconds.
        tmax: float
            Maximum time lag in seconds.
        direction: int
            Direction of the model. If ``direction == -1``,
            `tmin` and `tmax` are inverted

    Returns
    -------
        list of array-like:
            List of Features (i.e. x)
        list of array-like
            List of Estimands(i.e. y)
        tmin: float
            Minimum time lag.
        tmax: float
            Maximum time lag.
    """
    if direction == 1:
        x, y = stimulus, response
    elif direction == -1:
        x, y = response, stimulus
        tmin, tmax = -1 * tmax, -1 * tmin
    else:
        raise ValueError("Direction must be 1 or -1.")
    return x, y, tmin, tmax


def truncate(x: Array, min_idx: int, max_idx: int) -> Array:
    """
    Truncate matrix by rows (i.e. the time dimension in a TRF).

    Parameters
    ----------
    x: array-like
        Matrix to truncate.
    min_idx: int
        Smallest (time) index to include.
    max_idx: int
        Smallest (time) index to include.

    Returns
    -------
        array-like
            Truncated version of ``x``.
    """
    row_slice = slice(max(0, max_idx), min(0, min_idx) + len(x))
    x_truncated = x[row_slice]
    return x_truncated


def covariance_matrices(
    x: Union[Array, ArrayList],
    y: Union[Array, ArrayList],
    lags: List,
    zeropad: bool = True,
    preload: bool = True,
) -> Tuple[Array, Array]:
    """
    Compute the covariance of x and y and the autocovariance of x

    When passed a list of trials for x, and y, covariance
    matrices will be computed per trial and summed.

    Parameters
    ----------
    x: array-like or list of array-like
        Input data in samples-by-features array or list of such arrays.
    y: array-like or list of array-like
        Output data in samples-by-features array or list of such arrays.
    lags: list
        Time lags in samples.
    zeropad: bool
        If True (default), pad the input with zeros, if false, truncate the output.

    Returns
    -------
    array-like
        Three dimensional autocovariance matrix. 1st dimension's size is the number
        of trials, 2nd and 3rd dimensions' size is lags times features in x.
        If x contains only one trial, the first dimension is empty and will be removed.
    array-like
        Three dimensional x-y-covariance matrix. 1st dimension's size is the number
        of trials, 2nd dimension's size is lags times features in x and 3rd dimension's
        size is features in y. If y contains only one trial, the first dimension is
        empty and will be removed.
    """
    x, xpx = _check_data(x)
    y, xpy = _check_data(y)
    if not xpx == xpy:
        raise TypeError("x and y trials must be of the same type!")
    else:
        xp = xpx
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    cov_xx, cov_xy = 0, 0
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        x_lag = lag_matrix(x_i, lags, zeropad)
        if preload is True:
            if i == 0:
                cov_xx = xp.zeros((len(x), x_lag.shape[-1], x_lag.shape[-1]))
                cov_xy = xp.zeros((len(y), x_lag.shape[-1], y_i.shape[-1]))
            cov_xx[i] = x_lag.T @ x_lag
            cov_xy[i] = x_lag.T @ y_i
        else:
            cov_xx += x_lag.T @ x_lag
            cov_xy += x_lag.T @ y_i
    if preload is False:
        cov_xx, cov_xy = cov_xx / len(x), cov_xy / len(x)

    return cov_xx, cov_xy


def lag_matrix(
    x: Array, lags: List[int], zeropad: bool = True, bias: bool = True
) -> Array:
    """
    Construct a matrix with time lagged input features.
    See also 'lagGen' in mTRF-Toolbox github.com/mickcrosse/mTRF-Toolbox.

    Parameters
    ----------
        x: array-like
            Input data matrix of shape time x features
        lags: list
            List of time lags in samples
        zeropad: bool
            If True (default) apply zero paddinf to the colums with non-zero
            time lags to ensure causality. Otherwise, truncate the matrix.
        bias: bool
            If True (default), concatenate an array of ones to the left
            of the array to include a constant bias term in the regression.

    Returns
    -------
        array-like
            Matrix of time lagged inputs with shape times x number of lags * number
            of features (+1 if bias==True).
            If zeropad is False, the first dimension is truncated.
    """
    xp = array_api_compat.array_namespace(x)
    n_lags = len(lags)
    n_samples, n_variables = x.shape
    if max(lags) > n_samples:
        raise ValueError("The maximum lag can't be longer than the signal!")
    x_lag = xp.zeros((n_samples, n_variables * n_lags))

    for idx, lag in enumerate(lags):
        col_slice = slice(idx * n_variables, (idx + 1) * n_variables)
        if lag < 0:
            x_lag[0 : n_samples + lag, col_slice] = x[-lag:, :]
        elif lag > 0:
            x_lag[lag:n_samples, col_slice] = x[0 : n_samples - lag, :]
        else:
            x_lag[:, col_slice] = x

    if zeropad is False:
        x_lag = truncate(x_lag, lags[0], lags[-1])

    if bias is not False:
        x_lag = xp.concatenate([xp.ones((x_lag.shape[0], 1)), x_lag], 1)

    return x_lag


def regularization_matrix(
    size: int, xp: ModuleType, method: Literal["ridge", "banded", "tikhonov"] = "ridge"
) -> Array:
    """
    Generates a sparse regularization matrix for the specified method.

    Parameters
    ----------
    size: int
        Size of the regularization matrix.
    xp: module
        The array API namespace for generating the output matrix
    method: str
        Regularization method. Can be 'ridge', 'banded' or 'tikhonov'.

    Returns
    -------
    array-like
        regularization matrix for specified ``size`` and ``method``.
    """
    if method in ["ridge", "banded"]:
        regmat = xp.identity(size)
        regmat[0, 0] = 0
    elif method == "tikhonov":
        regmat = xp.identity(size)
        regmat -= 0.5 * (xp.diag(xp.ones(size - 1), 1) + xp.diag(xp.ones(size - 1), -1))
        regmat[1, 1] = 0.5
        regmat[size - 1, size - 1] = 0.5
        regmat[0, 0] = 0
        regmat[0, 1] = 0
        regmat[1, 0] = 0
    else:
        regmat = xp.zeros((size, size))
    return regmat


def banded_regularization(
    n_lags: int, coefficients: list, bands: List[int], xp: ModuleType
) -> Array:
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
    xp: module
        The array API namespace for generating the output matrix

    Returns
    -------
    array-like
        The banded regularization matrix.
    """
    if bands is None:
        raise ValueError("Must provide band sizes when using banded ridge regression!")
    if not len(bands) == len(coefficients):
        raise ValueError("Coefficients and bands must be of same size!")
    lag_coefs = []
    # repeat the coefficient for each occurence of the corresponding band
    for c, f in zip(coefficients, bands):
        lag_coefs.append(xp.repeat(c, f))
    lag_coefs = xp.concatenate(lag_coefs)
    # repeat that sequence for each lag
    diagonal = xp.concatenate([lag_coefs for i in range(n_lags)])
    diagonal = xp.concatenate([[0], diagonal])
    return xp.diag(diagonal)
