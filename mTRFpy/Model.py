# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""
import sklearn as skl
from sklearn.base import BaseEstimator,RegressorMixin, TransformerMixin

from . import DataStruct as ds
from . import Tools as tls
from . import Operations as op

DirEnum = tuple([-1,1]) 
    
class CTRF:
    
    def __init__(self):
        self.w = None
        self.b = None
        self.t = None
        self.Dir = None
        self.Type = 'multi'
        self.Zeropad = True
        self.fs = -1
        
    def train(self,stim,resp,Dir,fs,tmin_ms,tmax_ms,Lambda,**kwargs):
        assert Dir in DirEnum
        
        if (Dir == 1):
            x = stim
            y = resp
        else:
            x = resp
            y = stim
            tmin_ms, tmax_ms = Dir * tmax_ms, Dir * tmin_ms
        
        w,b,lags = tls.train(x,y,fs,tmin_ms,tmax_ms,Lambda,**kwargs)
        
        if kwargs.get('Type') != None:
            self.type = kwargs.get('Type')
        
        if kwargs.get('Zeropad') != None:
            self.Zeropad = kwargs.get('Zeropad')
            
        self.w, self.b = w, b
        self.Dir = Dir
        self.t = op.Idxs2msec(lags,fs)
        self.fs = fs
    
    def predict(self,stim,resp,**kwargs):
        assert self.Dir in DirEnum
        if self.Dir == 1:
            x = stim; y = resp
        else:
            x = resp; y = stim
        
        
        
    

class CSKlearnTRF(BaseEstimator,RegressorMixin, TransformerMixin, CTRF):
    '''
    main difference is that Dir will always be 1
    
    '''
    
    def __init__(self,fs, tmin_ms, tmax_ms, Lambda,**kwargs):
        super().__init__()
        self.Dir = 1
        self.fs = fs
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.Lambda = Lambda
        self.Type = 'multi'
        self.Zeropad = True
        self.kwargs = kwargs
        
    def fit(self,x,y):
        x = skl.utils.check_array(x)
        y = skl.utils.check_array(y)
        self.train(x,y,self.Dir,self.fs,self.tmin_ms,self.tmax_ms,self.Lambda,**self.kwargs)
    
    def predict(self,x):
        pass
    
    def transform(self,x):
        pass