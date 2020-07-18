# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:25 2020

@author: Jin Dou
"""
import numpy as np
from . import Operations as op
from . import DataStruct as ds


def truncFloat(a,digits):
    stepper = 10.0 ** digits
    return np.trunc(stepper * a) / stepper

def cmp2NArray(a,b,decimalNum = None):
    if decimalNum != None:
        a = np.around(a,decimalNum)
        b = np.around(b,decimalNum)
    return np.array_equal(a,b)

def olscovmat(x:ds.CDataList,y:ds.CDataList,lags,Type = 'multi',Zeropad = True ,Verbose = True):
    
    nXVar = x.nVar
    nYVar = y.nVar
    nLag = len(lags)
    
    assert Type in TypeEnum
#    Cxx = None
#    Cxy = None
#    
#    if Type == 'multi':
#        nXLagVar = nXVar*nLag + 1
#        Cxx = np.zeros((nXLagVar,nXLagVar))
#        Cxy = np.zeros((nXLagVar,nYVar))
#    else:
#        nXLagVar = nXVar + 1
#        Cxx = np.zeros((nXLagVar,nXLagVar,nLag))
#        Cxy = np.zeros((nXLagVar,nYVar,nLag))
    
    for f in range(x.fold):
        xLag = op.genLagMat(x[f],lags)
        CxxTemp = op.calCovariance(xLag,xLag.copy())
        CxyTemp = op.calCovariance(xLag,y[f])
        
        if Type == 'multi':
            Cxx += CxxTemp
            Cxy += CxyTemp
    
    return Cxx, Cxy

def train(x,y,fs,tmin_ms,tmax_ms,Lambda,**kwarg):
    if not isinstance(x, ds.CDataList):
        x = ds.CDataList(x)
    if not isinstance(y, ds.CDataList):
        y = ds.CDataList(y)
    
    assert x.fold == y.fold
    lags = op.msec2Idxs([-100,400],fs)
    Cxx,Cxy = olscovmat(x,y,lags,**kwarg)
    
    Delta = 1/fs
    RegM = op.genRegMat(Cxx.shape[1]) * Lambda / Delta
    wori = np.matmul(np.linalg.inv(Cxx + RegM), Cxy) / Delta
    b = wori[0]
    w = wori[1:].reshape((x.nVar,len(lags),y.nVar),order = 'F')
    return w,b,lags





    