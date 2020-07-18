# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""

from . import DataStruct as ds
from . import Tools as tls
from . import Operations as op
from sklearn.base import BaseEstimator,RegressorMixin, TransformerMixin


class CSKlearnTRF(BaseEstimator,RegressorMixin, TransformerMixin):
    
    def __init__(self):
        super().__init__()
        
    def fit(self,x,y):
        pass
    
    def predict(self,x):
        pass
    
    def transform(self,x):
        pass

DirEnum = tuple([-1,1]) 
    
class CTRF:
    
    def __init__(self):
        self.w = None
        self.b = None
        self.t = None
        self.Dir = None
        self.Type = 'multi'
        self.Zeropad = True
        
    def train(self,stim:ds.CDataList,resp:ds.CDataList,Dir,fs,tmin_ms,tmax_ms,Lambda,**kwarg):
        assert Dir in DirEnum
        
        if (Dir == 1):
            x = stim
            y = resp
        else:
            x = resp
            y = stim
            tmin_ms, tmax_ms = Dir * tmax_ms, Dir * tmin_ms
        
        w,b,lags = tls.train(x,y,fs,tmin_ms,tmax_ms,Lambda,**kwarg)
        
        if kwarg.get('Type') != None:
            self.type = kwarg.get('Type')
        
        if kwarg.get('Zeropad') != None:
            self.Zeropad = kwarg.get('Zeropad')
            
        self.w, self.b = w, b
        self.Dir = Dir
        self.t = op.Idxs2msec(lags,fs)
    
    def predict(self,):
        pass