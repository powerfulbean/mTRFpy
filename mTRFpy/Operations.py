# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:50:28 2020

@author: Jin Dou
"""

import numpy as np
from . import Protocols as prtcls


def calCovariance(x,y):
    '''
    calculat the covariance of two matrices
    x: left matrix
    y: right matrix
    
    if the input for x and y are both 1-D vectors, they will be reshaped to (len(vector),1)
    '''
    oPrtclsData = prtcls.CProtocolData()
    x,y = oPrtclsData(x,y)
    return np.matmul(x.T,y)


def genLagMat(x,lags,zeropad = True,bias =True): #
    '''
    build the lag matrix based on input.
    x: input matrix
    lags: a list (or list like supporting len() method) of integers, 
         each of them should indicate the time lag in samples.
    
    see also 'lagGen' in mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox
    
    
    #To Do:
       make warning when absolute lag value is bigger than the number of samples
    '''
    oPrtclsData = prtcls.CProtocolData()
    x = oPrtclsData(x)
    nLags = len(lags)
    
    nSamples = x.shape[0]
    nVar = x.shape[1]
    lagMatrix = np.zeros((nSamples,nVar*nLags))
    
    for idx,lag in enumerate(lags):
        colSlice = slice(idx * nVar,(idx + 1) * nVar)
        if lag < 0:
            lagMatrix[0:nSamples + lag,colSlice] = x[-lag:,:]
        elif lag > 0:
            lagMatrix[lag:nSamples,colSlice] = x[0:nSamples-lag,:]
        else:
            lagMatrix[:,colSlice] = x
    
    return lagMatrix

def genSmplIdxSeqByMsecRg(msecRange,fs):
    '''
    convert a millisecond range to a list of sample indexes
    
    the left and right ranges will both be included
    '''
    assert len(msecRange) == 2
    
    tmin = msecRange[0]/1e3
    tmax = msecRange[1]/1e3
    return list(range(int(np.floor(tmin*fs)),int(np.ceil(tmax*fs)) + 1))

def genRegMat(n, method = 'ridge'):
    '''
    generates a sparse regularization matrix of size (n,n) for the specified method.
    see also regmat.m in mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox
    '''
    regMatrix = None
    if method == 'ridge':
        regMatrix = np.identity(n)
        regMatrix[0,0] = 0
    elif method == 'Tikhonov':
        regMatrix = np.identity(n)
        regMatrix -= 0.5 * (np.diag(np.ones(n-1),1) + np.diag(np.ones(n-1),-1))
        regMatrix[1,1] = 0.5
        regMatrix[n-1,n-1] = 0.5
        regMatrix[0,0] = 0
        regMatrix[0,1] = 0
        regMatrix[1,0] = 0
    else:
        regMatrix = np.zeros((n,n))
    return regMatrix