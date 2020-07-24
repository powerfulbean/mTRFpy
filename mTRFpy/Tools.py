# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:25 2020

@author: Jin Dou
"""
import numpy as np
from . import Operations as op
from . import DataStruct as ds
from . import Protocols as pt


oDataPrtcl = pt.CProtocolData()

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
    b = wori[0:1]
    w = wori[1:].reshape((x.nVar,len(lags),y.nVar),order = 'F')
    return w,b,lags


#def predict(*arg,windowSize:int):
#    assert windowSize >= 0
#    if windowSize:
#        nWin = sum(np.floor([len(n)/windowSize for n in nYObs/windowSize]))
#    else:
#        nWin = nFold
    

DimEnum = [0,1]
CorrEnum = ['Pearson','Spearman']
def evaluate(y,pred,dim:int = 0, corr = 'Pearson',error='mse',window = 0):
    '''
    to do:
        implement corr = 'Spearman'
    '''
    assert dim in DimEnum
    assert corr in CorrEnum
    assert window >= 0
    y,pred = oDataPrtcl(y,pred)
    
    if dim == 1:
        y = y.T
        pred = pred.T
        
    nYObs,nYVar = y.shape[0:2]
    nPObs,nPVar = pred.shape[0:2]
    assert nYObs == nPObs and nYVar == nPVar
    
    nWin = 0
    if window:
        nWin = int(np.floor(nYObs/window))
    else:
        nWin = 1
    
    
    r = list()
    err = list()
    
    for i in range(nWin):
        if window:
            idx = slice(i * window,window * (i+1))
            yi = y[idx]
            pi = pred[idx]
        else:
            yi = y
            pi = pred
    
        rTemp = op.pearsonr(yi,pi)
        errTemp = op.error(yi,pi)
        r.append(rTemp)
        err.append(errTemp)
        
    return r,err
    
    
    
    