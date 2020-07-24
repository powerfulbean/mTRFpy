# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:44:37 2020

@author: Jin Dou
"""
import sys
sys.path.append('../../')
from StellarBrainwav.DataStruct.DataSet import CDataRecord
from StellarInfra.DirManage import CDirectoryConfig
from StellarInfra.StageControl import CStageControl
from mTRFpy import InterfaceMatlab as ifMatlab
from mTRFpy import Operations as op
from mTRFpy import Tools as tls
from mTRFpy import DataStruct as ds
from mTRFpy import Model as md
import numpy as np

oDir = CDirectoryConfig(['TestData'],'mTRFpy.conf')

oStage = CStageControl([2,2.1,2.2])

if oStage(0):
    '''
    test mtrfTrain procedure
    '''
    temp = ifMatlab.loadMatFile(oDir.TestData + 'speech_data.mat')
    
    resp = temp['resp'] * temp['factor'][0,0]
    stim = temp['stim']
    fs = temp['fs'][0,0]

if oStage(1):
    oData = CDataRecord(resp,stim,list(),fs)
    
    lags = op.msec2Idxs([-100,400],oData.srate)
    stimLag = op.genLagMat(oData.stimuli,lags)
    Cxx = op.calCovariance(stimLag,stimLag.copy())
    Cxy = op.calCovariance(stimLag,oData.data)
    
    Delta = 1/oData.srate
    Lambda = 0.1
    RegM = op.genRegMat(stimLag.shape[1]) * Lambda / Delta
    wori = np.matmul(np.linalg.inv(Cxx + RegM), Cxy) / Delta
    b = wori[0]
    w = wori[1:].reshape((oData.stimuli.shape[1],len(lags),oData.data.shape[1]),order = 'F')
    
    
    
    model_benchmark = ifMatlab.loadMatFile(oDir.TestData + 'model_benchmark.mat')
    w_benchmark = model_benchmark['w']
    print(tls.cmp2NArray(w,w_benchmark,8))

if oStage(1.1):
    lags = op.msec2Idxs([-100,400],fs)
    x = stim
    y = resp
    if not isinstance(x, ds.CDataList):
        x = ds.CDataList(x)
    if not isinstance(y, ds.CDataList):
        y = ds.CDataList(y)
        
#    Cxx_test,Cxy_test = tls.olscovmat(x+x.copy(),y+y.copy(),lags)
    w_11,b_11,lags_11 = tls.train(stim,resp,fs,-100,400,Lambda)
    print(tls.cmp2NArray(w_11,w_benchmark,8),tls.cmp2NArray(b_11,model_benchmark['b'],8),\
          tls.cmp2NArray(lags_11,lags,8))


if oStage(2):
    '''
    test mtrfTrain and mtrfPred 
    '''
    temp = ifMatlab.loadMatFile(oDir.TestData + 'trainNtest_bench.mat')
    bench_key_list = ['w','b','t','fs','Dir','type']
    Model_Bench_Dict = dict()
    for key in bench_key_list:
        Model_Bench_Dict[key] = temp[key]

if oStage(2.1):
    '''
    Train
    '''
    respTrain = [d[0] for d in temp['resptrain']]
    stimTrain = [d[0] for d in temp['stimtrain']]
    fs = temp['fs'][0,0]
    Dir = temp['Dir'][0,0]
    
    model = md.CTRF()
    model.train(stimTrain,respTrain,Dir,fs,0,250,100,Zeropad = False)
    
    
    
    '''
    validate the result
    '''
    w_test = model.w
    b_test = model.b
    print(tls.cmp2NArray(w_test,np.expand_dims(Model_Bench_Dict['w'],2),12))
    
if oStage(2.2):
    '''
    Predict
    '''
    
    respTest = temp['resptest']
    stimTest = temp['stimtest']
    windowSize_ms = 0
    
    if model.Dir == 1:
        x = stimTest; y = respTest
    else:
        x = respTest; y = stimTest
    
    if not isinstance(x, ds.CDataList):
        x = ds.CDataList(x)
    if not isinstance(y, ds.CDataList):
        y = ds.CDataList(y)
    
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
        
        if Type == 'multi':
            predTemp = np.matmul(xLag,w)
            pred.append(predTemp)
            
            if y != None:
                yTrunc = op.truncate(y[i],lags[0],lags[-1]) 
                rTempList,errTempList = tls.evaluate(yTrunc,predTemp)
                r.extend(rTempList)
                err.extend(errTempList)
    '''
    validate the result
    '''
    
    print(tls.cmp2NArray(pred[0],temp['pred'],11))
    rCell = ds.CDataList(r)
    errCell = ds.CDataList(err)
    print(tls.cmp2NArray(rCell[0],temp['test_r'],14),\
          tls.cmp2NArray(errCell[0],temp['test_err'],15))
    
    
    
if oStage(3):
    temp = np.array([[1,2,3],[1,2,3],[1,2,3]])
    oList = ds.CDataList(temp,2,2)
    
    
if oStage(4):
    '''
    test CSKlearnTRF
    '''
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    
    oScaler = MinMaxScaler()
    model = md.CSKlearnTRF(64,0,250,100,Zeropad = False)
    clf = Pipeline([('minmax',oScaler),
                    ('TRF',model)])
    clf.fit(respTrain[0],stimTrain[0])
    
    
    
    
    


#temp1 = ifMatlab.saveMatFile( oDir.TestData+ 'speech_data_out.mat',temp)
