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
    output = ds.DataListOp(op.calOlsCovMat)(x,y,lags,Type,Zeropad)
    CxxList = [c[0] for c in output]
    CxyList = [c[1] for c in output]
    Cxx,Cxy = sum(CxxList),sum(CxyList)
    return Cxx, Cxy

def train(x,y,fs,tmin_ms,tmax_ms,Lambda,**kwarg):
    if not isinstance(x, ds.CDataList):
        x = ds.CDataList(x)
    if not isinstance(y, ds.CDataList):
        y = ds.CDataList(y)
    
    assert x.fold == y.fold
    lags = op.msec2Idxs([tmin_ms,tmax_ms],fs)
    Cxx,Cxy = olscovmat(x,y,lags,**kwarg)
    
    Delta = 1/fs
    RegM = op.genRegMat(Cxx.shape[1]) * Lambda / Delta
    wori = np.matmul(np.linalg.inv(Cxx + RegM), Cxy) / Delta
    b = wori[0]
    w = wori[1:].reshape((x.nVar,len(lags),y.nVar),order = 'F')
    return w,b,lags





    