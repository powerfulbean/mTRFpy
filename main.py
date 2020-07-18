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

oStage = CStageControl([1,1.1])

if oStage(1):
    '''
    test mtrfTrain procedure
    '''
    temp = ifMatlab.loadMatFile(oDir.TestData + 'speech_data.mat')
    
    resp = temp['resp'] * temp['factor'][0,0]
    stim = temp['stim']
    fs = temp['fs'][0,0]
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
    w_11,b_11,lags_11 = tls.train(stim,resp,fs,-100,400,Lambda)


if oStage(2):
    '''
    test mtrfTrain and mtrfPred 
    '''
    temp = ifMatlab.loadMatFile(oDir.TestData + 'trainNtest_bench.mat')
    
    respTrain = [d[0] for d in temp['resptrain']]
    stimTrain = [d[0] for d in temp['stimtrain']]
    
    bench_key_list = ['w','b','t','fs','Dir','type']
    Model_Bench_Dict = dict()
    for key in bench_key_list:
        Model_Bench_Dict[key] = temp[key]
    
    
if oStage(3):
    temp = np.array([[1,2,3],[1,2,3],[1,2,3]])
    oList = ds.CDataList(temp,2,2)
    


#temp1 = ifMatlab.saveMatFile( oDir.TestData+ 'speech_data_out.mat',temp)
