# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:25 2020

@author: Jin Dou
"""
import numpy as np
from . import Core
from . import DataStruct as ds
from .DataStruct import CDataList
from . import Protocols as pt
# from memory_profiler import profile

oDataPrtcl = pt.CProtocolData()
oStimRespPrtcl = pt.CStimRespProtocol()

def truncFloat(a,digits):
    stepper = 10.0 ** digits
    return np.trunc(stepper * a) / stepper

# @profile
def olscovmat(x:ds.CDataList,y:ds.CDataList,lags,Type = 'multi',Zeropad = True ,Verbose = True):
    output = ds.DataListOp(Core.calOlsCovMat)(x,y,lags,Type,Zeropad)
    Cxx = output[0]
    Cxy = output[1]
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
    lags = Core.msec2Idxs([tmin_ms,tmax_ms],fs)
    Cxx,Cxy = olscovmat(x,y,lags,**kwarg)
    # print('tls train, start regularization matrix')
    Delta = 1/fs
    RegM = Core.genRegMat(Cxx.shape[1]) * Lambda / Delta
    if oCuda is None:
        wori = np.matmul(np.linalg.inv(Cxx + RegM), Cxy) / Delta
    else:
        RegM = oCuda.cp.asarray(RegM)
        Cxx = oCuda.cp.array(Cxx)
        Cxy = oCuda.cp.array(Cxy)
            
        wori = oCuda.cp.matmul(oCuda.cp.linalg.inv(Cxx + RegM), Cxy) #/ Delta
        oCuda.cp.cuda.Stream.null.synchronize()
        # print(type(wori),type(Cxx),type(Cxy))
        wori = oCuda.cp.asnumpy(wori)
        wori = wori / Delta
        del Cxx
        del Cxy
        oCuda.memPool.free_all_blocks()
    # print('tls train, regularization finish')
    # print(type(wori),wori.shape)
    wori = np.asarray(wori)
    # wori = wori.toarray()
    b = wori[0:1]
    w = wori[1:].reshape((x.nVar,len(lags),y.nVar),order = 'F')
    # print('tls train finish')
    return w,b,lags


def predict(model,x:CDataList,y=0,windowSize_ms:int = 0,zeropad:bool = True,dim = 0, specifyLag:int = -1, specifyChan:int = -1):
    # assert windowSize >= 0
    # if windowSize:
    #     nWin = sum(np.floor([len(n)/windowSize for n in nYObs/windowSize]))
    # else:
    #     nWin = nFold
    Core.sparseFlag = False
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
    
    lags = Core.msec2Idxs([model.t[0],model.t[-1]],model.fs)
    windowSize = round(windowSize_ms * model.fs)
    
    Type = model.Type
    
    delta = 1/model.fs
    
#    assert Type
    if model.Type == 'multi':
        w = model.w.copy()
        if specifyLag >= 0:
            lags = [lags[specifyLag]]
            w = w[:,specifyLag:specifyLag+1,:]
            # print(specifyLag,specifyLag+1,w.shape,lags[0],lags[0]+1,model.w.shape)
        if specifyChan >= 0:
            w = w[specifyChan:specifyChan+1,:,:]
            nXVar = 1
            # print(x[0].shape)
            x = x.getChan(specifyChan)
            # print(x[0].shape,w.shape)
        # print(nXVar,lags)
        w = np.concatenate([model.b,w.reshape((nXVar*len(lags),nYVar),order = 'F')])*delta
    else:
        w = 1
    
    pred = ds.CDataList()
    r = list()
    err = list()
    for i in range(x.fold):
        xLag = Core.genLagMat(x[i],lags,model.Zeropad)
        # print('\rtest fold: ',i,end='\r')
        if Type == 'multi':
            # print(xLag.shape,w.shape)
            # predTemp = np.matmul(xLag,w)
            # print(xLag.shape,lags)
            predTemp = Core.matMul(xLag,w)
            # print(predTemp.shape)
            pred.append(predTemp)
            
            if y != None:
                if not zeropad:
                    yTrunc = Core.truncate(y[i],lags[0],lags[-1])
                else:
                    yTrunc = y[i]
                rTempList,errTempList = evaluate(yTrunc,predTemp,dim=dim)
                r.extend(rTempList)
                err.extend(errTempList)
        del xLag
    # print('\n')
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
    
        rTemp = Core.pearsonr(yi,pi)
        errTemp = Core.error(yi,pi)
        r.append(rTemp)
        err.append(errTemp)
        
    return np.array(r),np.array(err)

def split(stim,resp,nFold):
    outputStim = list()
    outputResp = list()
    stim,resp = oStimRespPrtcl(stim,resp)
    lenSeg = int(np.ceil(len(stim)/nFold))
    nObs = stim.shape[0]
    for i in range(nFold):
        rowSlice = slice(i * lenSeg,min((i+1) * lenSeg,nObs))
        outputStim.append(stim[rowSlice])
        outputResp.append(resp[rowSlice])
    outputStim = outputStim
    outputResp = outputResp
    return CDataList(outputStim),CDataList(outputResp)

def partition(stim,resp,nFold,testFold):
    outputStim,outputResp = split(stim, resp, nFold)
    r1,s1,r2,s2 = CDataList(outputStim[:-testFold]), CDataList(outputResp[:-testFold]), \
            CDataList(outputStim[-testFold:]), CDataList(outputResp[-testFold:])
                
    return r1,s1,r2,s2
    
    
