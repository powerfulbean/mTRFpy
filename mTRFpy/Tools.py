# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:25 2020

@author: Jin Dou
"""
import numpy as np
from . import Operations as op
from . import DataStruct as ds
from . import Protocols as pt
# from memory_profiler import profile

oDataPrtcl = pt.CProtocolData()

def truncFloat(a,digits):
    stepper = 10.0 ** digits
    return np.trunc(stepper * a) / stepper

def cmp2NArray(a,b,decimalNum = None):
    if decimalNum != None:
        a = np.around(a,decimalNum)
        b = np.around(b,decimalNum)
    return np.array_equal(a,b)

# @profile
def olscovmat(x:ds.CDataList,y:ds.CDataList,lags,Type = 'multi',Zeropad = True ,Verbose = True):
    output = ds.DataListOp(op.calOlsCovMat)(x,y,lags,Type,Zeropad)
    # print('olscovmat collect data')
    # CxxList = [c[0] for c in output]
    # CxyList = [c[1] for c in output]
    # Cxx,Cxy = sum(CxxList),sum(CxyList)
    
    # output = np.sum(output,axis = 0)
    Cxx = output[0]
    Cxy = output[1]
    # print('olscovmat collect data finish')
    return Cxx, Cxy

# @profile
def train(x,y,fs,tmin_ms,tmax_ms,Lambda,oCuda = None,**kwarg):
    if not isinstance(x, ds.CDataset):
        if not isinstance(x, ds.CDataList):
            x = ds.CDataList(x)
    if not isinstance(y, ds.CDataset):
        if not isinstance(y, ds.CDataList):
            y = ds.CDataList(y)
    
    assert x.fold == y.fold
    lags = op.msec2Idxs([tmin_ms,tmax_ms],fs)
    Cxx,Cxy = olscovmat(x,y,lags,**kwarg)
    print('tls train, start regularization matrix')
    Delta = 1/fs
    RegM = op.genRegMat(Cxx.shape[1]) * Lambda / Delta
    if oCuda is None:
        wori = np.matmul(np.linalg.inv(Cxx + RegM), Cxy) / Delta
    else:
        RegM = oCuda.cp.asarray(RegM)
        Cxx = oCuda.cp.asarray(Cxx)
        Cxy = oCuda.cp.asarray(Cxy)
        wori = oCuda.cp.matmul(oCuda.cp.linalg.inv(Cxx + RegM), Cxy) #/ Delta
        oCuda.cp.cuda.Stream.null.synchronize()
        # print(type(wori),type(Cxx),type(Cxy))
        wori = oCuda.cp.asnumpy(wori)
        wori = wori / Delta
        del Cxx
        del Cxy
        oCuda.memPool.free_all_blocks()
    print('tls train, regularization finish')
    b = wori[0:1]
    w = wori[1:].reshape((x.nVar,len(lags),y.nVar),order = 'F')
    print('tls train finish')
    return w,b,lags


def predict(model,x,y=0,windowSize_ms:int = 0,zeropad:bool = True):
    # assert windowSize >= 0
    # if windowSize:
    #     nWin = sum(np.floor([len(n)/windowSize for n in nYObs/windowSize]))
    # else:
    #     nWin = nFold
    nXObs = [len(d) for d in x]
    nXVar = x.nVar
    if y == None:
        nYObs = nXObs
        nYVar = model.w.shape[2]
    else:
        nYObs = [len(d) for d in y]
        nYVar = y.nVar
    nFold = x.fold
    
    for idx,n in enumerate(nXObs):
        assert n == nYObs[idx]
    
    lags = op.msec2Idxs([model.t[0],model.t[-1]],model.fs)
    windowSize = round(windowSize_ms * model.fs)
    
    Type = model.Type
    
    delta = 1/model.fs
    
#    assert Type
    if model.Type == 'multi':
        w = model.w.copy()
        w = np.concatenate([model.b,w.reshape((nXVar*len(lags),nYVar),order = 'F')])*delta
    else:
        w = 1
    
    pred = ds.CDataList()
    r = list()
    err = list()
    cursor = 0
    for i in range(x.fold):
        xLag = op.genLagMat(x[i],lags,model.Zeropad)
        print('\rtest fold: ',i,end='\r')
        if Type == 'multi':
            predTemp = np.matmul(xLag,w)
            # print(predTemp.shape)
            pred.append(predTemp)
            
            if y != None:
                if not zeropad:
                    yTrunc = op.truncate(y[i],lags[0],lags[-1])
                else:
                    yTrunc = y[i]
                rTempList,errTempList = evaluate(yTrunc,predTemp)
                r.extend(rTempList)
                err.extend(errTempList)
        print('\n')
    if y == None:
        return pred
    else:
        return pred,np.array(r),np.array(err)

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
        
    return np.array(r),np.array(err)
    
    
    
    