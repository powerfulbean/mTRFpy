# -*- coding: utf-8 -*-
"""
Created on Sat May  7 00:01:54 2022

@author: Jin Dou
"""

'''
replicating the example provided in mTRFToolbox:
    https://github.com/mickcrosse/mTRF-Toolbox
    
'''

from mTRFpy import Tools
from mTRFpy.Model import TRF
from mTRFpy.Tools import cmp2NArray

oStage = Tools.StageControl([1,2])

if oStage(1):
    #simple encoder validation
    speechAndRespData = Tools.loadMatFile('speech_data.mat')
    encoderResult = Tools.loadMatFile('EncoderTask.mat')
    stim = speechAndRespData['stim']
    resp = speechAndRespData['resp']
    fs = speechAndRespData['fs'][0,0]
    oTRFEncoder = TRF()
    oTRFEncoder.train(stim,resp,1,fs,-100,200,100)
    assert cmp2NArray(oTRFEncoder.weights,encoderResult['modelEncoder']['w'][0,0],10)
    assert cmp2NArray(oTRFEncoder.bias,encoderResult['modelEncoder']['b'][0,0],12)
    predE,rE,errE = oTRFEncoder.predict(stim,resp)
    assert cmp2NArray(predE[0], encoderResult['predResp'],10)
    assert cmp2NArray(rE, encoderResult['predRespStats']['r'][0,0],11)
    assert cmp2NArray(errE, encoderResult['predRespStats']['err'][0,0],13)

if oStage(2):
    #simple decoder validation
    speechAndRespData = Tools.loadMatFile('speech_data.mat')
    decoderResult = Tools.loadMatFile('DecoderTask.mat')
    stim = speechAndRespData['stim']
    resp = speechAndRespData['resp']
    fs = speechAndRespData['fs'][0,0]
    oTRFDecoder = TRF()
    oTRFDecoder.train(stim,resp,-1,fs,-100,200,100)
    assert cmp2NArray(oTRFDecoder.weights,decoderResult['modelDecoder']['w'][0,0],8)
    assert cmp2NArray(oTRFDecoder.bias,decoderResult['modelDecoder']['b'][0,0],11)
    predD,rD,errD = oTRFDecoder.predict(stim,resp)
    assert cmp2NArray(predD[0], decoderResult['predStim'],8)
    assert cmp2NArray(rD, decoderResult['predStimStats']['r'][0,0],12)
    assert cmp2NArray(errD, decoderResult['predStimStats']['err'][0,0],16)

