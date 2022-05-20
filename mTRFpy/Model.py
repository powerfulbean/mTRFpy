# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""
import sklearn as skl
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import ShuffleSplit, LeaveOneOut
from multiprocessing import Pool, shared_memory
from . import Tools
from . import DataStruct as ds
from . import Basics as bs
from . import Core
import sys
from tqdm import tqdm


def funcTrainNVal(oInput):
    trainIdx = oInput[0]
    testIdx = oInput[1]
    idx = oInput[2]
    nSplits = oInput[3]
    modelParam = shared_memory.ShareableList(name='sharedModelParam')
    Dir = modelParam[0]
    fs = modelParam[1]
    tmin_ms = modelParam[2]
    tmax_ms = modelParam[3]
    Lambda = modelParam[4]
    stim = modelParam[5]
    resp = modelParam[6]
    oTRF = TRF()
    stimTrain = stim.selectByIndices(trainIdx)
    respTrain = resp.selectByIndices(trainIdx)
    oTRF.train(stimTrain, respTrain, Dir, fs, tmin_ms, tmax_ms, Lambda)
    stimTest = stim.selectByIndices(testIdx)
    respTest = resp.selectByIndices(testIdx)
    _, r, err = oTRF.predict(stimTest, respTest)
    sys.stdout.write(
        "\r" + f"cross validation >>>>>..... split {idx}/{nSplits}\r")
    return (r, err)


def createSharedNArray(a, name):
    shm = shared_memory.SharedMemory(name=name, create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    return b


def crossVal(stim: ds.CDataList, resp: ds.CDataList,
             Dir, fs, tmin_ms, tmax_ms, Lambda,
             random_state=42, nWorkers=1, n_Splits=10, **kwargs):
    if np.isscalar(Lambda):
        r, err = crossValPerLambda(
                stim, resp, Dir, fs, tmin_ms, tmax_ms, Lambda,
                random_state=random_state, nWorkers=nWorkers,
                n_Splits=n_Splits, **kwargs)

        return (np.mean(r, axis=0, keepdims=True),
                np.mean(err, axis=0, keepdims=True))

    else:
        result = []
        for la in tqdm(Lambda, desc='lambda', leave=False):
            r, err = crossValPerLambda(
                    stim, resp, Dir, fs, tmin_ms, tmax_ms, la,
                    random_state=random_state, nWorkers=nWorkers,
                    n_Splits=n_Splits, **kwargs)
            result.append(
                (np.mean(r, axis=0, keepdims=True),
                 np.mean(err, axis=0, keepdims=True))
            )
        return result


def crossValPerLambda(stim: ds.CDataList, resp: ds.CDataList,
                      Dir, fs, tmin_ms, tmax_ms, Lambda,
                      random_state=42, nWorkers=1, n_Splits=10, **kwargs):
    stim = ds.CDataList(stim)
    resp = ds.CDataList(resp)
    if n_Splits is not None:
        nSplits = n_Splits
        testSize = 1/nSplits
        if testSize > 0.1:
            testSize = 0.1
        rs = ShuffleSplit(n_splits=nSplits, test_size=testSize,
                          random_state=random_state)
    else:
        rs = LeaveOneOut()
        nStim = len(stim)
        nSplits = nStim

    if nWorkers <= 1:
        finalR = []
        finalErr = []
        idx = 0

        for trainIdx, testIdx in tqdm(rs.split(stim), desc='cv', leave=False):
            # print('def\rabc')
            # sys.stdout.write(f"cross validation >>>>>..... split {idx+1}/{nStim}")
            # sys.stdout.flush()
            # sys.stdout.write("\033[F")
            # sys.stdout.write("\033[K")
            # print("\r" + f"Lambda: {Lambda}; cross validation >>>>>..... split {idx+1}/{nSplits}",end='\r')

            idx += 1
            oTRF = TRF()
            stimTrain = stim.selectByIndices(trainIdx)
            respTrain = resp.selectByIndices(trainIdx)
            oTRF.train(stimTrain, respTrain, Dir, fs,
                       tmin_ms, tmax_ms, Lambda, **kwargs)

            stimTest = stim.selectByIndices(testIdx)
            respTest = resp.selectByIndices(testIdx)
            _, r, err = oTRF.predict(stimTest, respTest)
            finalR.append(r)
            finalErr.append(err)
    else:
        splitParam = []
        idx = 1
        for trainIdx, testIdx in rs.split(stim):
            splitParam.append([trainIdx, testIdx, idx, nSplits])
            idx += 1

        modelParam = shared_memory.ShareableList(
            [Dir, fs, tmin_ms, tmax_ms, Lambda],
            name='sharedModelParam')

        sharedStim = createSharedNArray(stim, 'sharedStim')
        sharedResp = createSharedNArray(resp, 'sharedResp')

        # stop
        with Pool(nWorkers) as pool:
            out = pool.imap(funcTrainNVal, splitParam,
                            chunksize=int(nSplits/nWorkers))
            finalR = [i[0] for i in out]
            finalErr = [i[1] for i in out]
    finalR = np.concatenate(finalR)
    finalErr = np.concatenate(finalErr)
    return finalR, finalErr  # np.mean(finalR,axis=0),np.mean(finalErr,axis=0)


class TRF:
    '''
    Class for the (multivariate) temporal response function.
    Can be used as a forward encoding model (stimulus to neural response)
    or backward decoding model (neural response to stimulus) using time lagged
    input features as per Crosse et al. (2016).
    Arguments:
        direction (int): Direction of the model. Can be 1 to fit a forward
            model (default) or -1 to fit a backward model.
        kind (str): Kind of model to fit. Can be 'multi' (default) to fit
            a multi-lag model using all time lags simulatneously or
            'single' to fit separate sigle-lag models for each individual lag.
        zeropad (bool): If True (defaul), pad the outer rows of the design
            matrix with zeros. If False, delete them.
    '''
    def __init__(self, direction=1, kind='multi', zeropad=True, method=''):
        self.weights = None
        self.bias = None
        self.times = None
        if direction in [1, -1]:
            self.direction = direction
        else:
            raise ValueError('Parameter direction must be either 1 or -1!')
        if kind in ['multi', 'single']:
            self.kind = kind
        else:
            raise ValueError(
                    'Paramter kind must be either "multi" or "single"!')
        if isinstance(zeropad, bool):
            self.zeropad = True
        else:
            raise ValueError('Parameter zeropad must be boolean!')
        self.fs = -1

    def __radd__(self, trf):
        if trf == 0:
            return self.copy()
        else:
            return self.__add__(trf)

    def __add__(self, trf):
        if not isinstance(trf, TRF):
            raise TypeError('Can only add to another TRF instance!')
        if not (self.direction == trf.direction) and (self.kind == trf.kind):
            raise ValueError('Added TRFs must be of same kind and direction!')
        trf_new = self.copy()
        trf_new.weights += trf.weights
        trf_new.bias += trf.bias
        return trf_new

    def __truediv__(self, num):
        trf_new = self.copy()
        trf_new.weights /= num
        trf_new.bias /= num
        return trf_new

    def train(self, stim, resp, Dir, fs, tmin_ms, tmax_ms, Lambda, **kwargs):

        if self.direction == 1:
            x, y = stim, resp
        elif self.direction == -1:
            x, y = resp, stim
            tmin_ms, tmax_ms = -1 * tmax_ms, -1 * tmin_ms

        w, b, lags = self._train(x, y, fs, tmin_ms, tmax_ms, Lambda, **kwargs)

        self.weights, self.bias = w, b
        self.direction = Dir
        self.times = Core.Idxs2msec(lags, fs)
        self.fs = fs

    def _train(self, x, y, fs, tmin, tmax, Lambda, **kwarg):

        tmin, tmax = tmin/1e3, tmax/1e3  # TODO: change time to seconds
        lags = list(range(int(np.floor(tmin*fs)), int(np.ceil(tmax*fs)) + 1))
        cov_xx, cov_xy = covariance_matrices(x, y, lags)
        delta = 1/fs
        regmat = regularization_matrix(cov_xx.shape[1], 'ridge')
        regmat *= Lambda / delta

        wori = np.matmul(np.linalg.inv(cov_xx + regmat), cov_xy) / delta
        b = wori[0:1]
        w = wori[1:].reshape((x.shape[1], len(lags), y.shape[1]), order='F')
        # print('tls train finish')
        return w, b, lags

    def predict(self, stim, resp=None, **kwargs):
        if isinstance(stim, np.ndarray):
            stim = ds.CDataList([stim])

        if isinstance(resp, np.ndarray):
            resp = ds.CDataList([resp])

        if self.direction == 1:
            x = stim
            y = resp
        else:
            x = resp
            y = stim

        return bs.predict(self, x, y, zeropad=self.zeropad, **kwargs)

    def save(self, path, name):
        output = dict()
        for i in self.__dict__:
            output[i] = self.__dict__[i]

        Tools.saveObject(output, path, name, '.mtrf')

    def load(self, path):
        temp = Tools.loadObject(path)
        for i in temp:
            setattr(self, i, temp[i])

    def copy(self):
        oTRF = TRF()
        for k, v in self.__dict__.items():
            value = v
            if getattr(v, 'copy', None) is not None:
                value = v.copy()
            setattr(oTRF, k, value)
        return oTRF

    def plotWeights(self, vecNames=None, ylim=None, newFig=True, chan=[None]):
        '''desined for models trained with combined vector '''
        from matplotlib import pyplot as plt
        times = self.times
        out = list()
        if self.direction == 1:
            nStimChan = self.weights.shape[0]
        elif self.direction == -1:
            nStimChan = self.weights.shape[-1]
        else:
            raise ValueError

        for i in range(nStimChan):
            if self.direction == 1:
                # take mean along the input dimension
                weights = self.weights[i, :, :]
            elif self.direction == -1:
                weights = self.weights[:, :, i].T
            else:
                raise ValueError
            if newFig:
                fig1 = plt.figure()
            else:
                fig1 = None
            plt.plot(times, weights[:, slice(*chan)])
            plt.title(vecNames[i] if vecNames is not None else '')
            plt.xlabel("time (ms)")
            plt.ylabel("a.u.")
            if ylim:
                plt.ylim(ylim)
            if fig1:
                out.append(fig1)
        return out


# define matrix operations
def truncate(x, tminIdx, tmaxIdx):
    '''
    the left and right ranges will both be included
    '''
    rowSlice = slice(max(0, tmaxIdx), min(0, tminIdx) + len(x))
    output = x[rowSlice]
    return output


def covariance_matrices(x, y, lags, zeropad=True):
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    x_lag = lag_matrix(x, lags, zeropad)
    cov_xx = x_lag.T @ x_lag
    cov_xy = x_lag.T @ y
    return cov_xx, cov_xy


def lag_matrix(x, lags, zeropad=True, bias=True):
    '''
    build the lag matrix based on input.
    x: input matrix
    lags: a list (or list like supporting len() method) of integers,
         each of them should indicate the time lag in samples.
    see also 'lagGen' in mTRF-Toolbox github.com/mickcrosse/mTRF-Toolbox
    #To Do:
       make warning when absolute lag value is bigger than the number of
       samples implement the zeropad part
    '''
    n_lags = len(lags)
    n_samples, n_variables = x.shape
    lag_matrix = np.zeros((n_samples, n_variables*n_lags))

    for idx, lag in enumerate(lags):
        col_slice = slice(idx * n_variables, (idx + 1) * n_variables)
        if lag < 0:
            lag_matrix[0:n_samples + lag, col_slice] = x[-lag:, :]
        elif lag > 0:
            lag_matrix[lag:n_samples, col_slice] = x[0:n_samples-lag, :]
        else:
            lag_matrix[:, col_slice] = x

    if zeropad is False:
        lag_matrix = truncate(lag_matrix, lags[0], lags[-1])

    if bias:
        lag_matrix = np.concatenate(
                [np.ones((lag_matrix.shape[0], 1)), lag_matrix], 1)

    return lag_matrix


def regularization_matrix(size, method='ridge'):
    '''
    generates a sparse regularization matrix for the specified method.
    see also regmat.m in https://github.com/mickcrosse/mTRF-Toolbox.
    '''
    if method == 'ridge':
        regmat = np.identity(size)
        regmat[0, 0] = 0
    elif method == 'Tikhonov':
        regmat = np.identity(size)
        regmat -= 0.5 * (np.diag(np.ones(size-1), 1) +
                         np.diag(np.ones(size-1), -1))
        regmat[1, 1] = 0.5
        regmat[size-1, size-1] = 0.5
        regmat[0, 0] = 0
        regmat[0, 1] = 0
        regmat[1, 0] = 0
    else:
        regmat = np.zeros((size, size))
    return regmat
