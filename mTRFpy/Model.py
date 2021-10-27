# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""
import sklearn as skl
from sklearn.base import BaseEstimator,RegressorMixin, TransformerMixin

from StellarInfra import IO as siIO
from . import DataStruct as ds
from . import Basics as bs
from . import Core
DirEnum = tuple([-1,1]) 
    
class CTRF:
    
    def __init__(self,):
        self.w = None
        self.b = None
        self.t = None
        self.Dir = None
        self.Type = 'multi'
        self.Zeropad = True
        self.fs = -1
        self._oCuda = None
        
    def train(self,stim,resp,Dir,fs,tmin_ms,tmax_ms,Lambda,**kwargs):
        assert Dir in DirEnum
        
        if (Dir == 1):
            x = stim
            y = resp
        else:
            x = resp
            y = stim
            tmin_ms, tmax_ms = Dir * tmax_ms, Dir * tmin_ms
        
        w,b,lags = bs.train(x,y,fs,tmin_ms,tmax_ms,Lambda,oCuda = self._oCuda,**kwargs)
        
        if kwargs.get('Type') != None:
            self.type = kwargs.get('Type')
        
        if kwargs.get('Zeropad') != None:
            self.Zeropad = kwargs.get('Zeropad')
            
        self.w, self.b = w, b
        self.Dir = Dir
        self.t = Core.Idxs2msec(lags,fs)
        self.fs = fs
    
    def predict(self,stim,resp = None,**kwargs):
        assert self.Dir in DirEnum
        if self.Dir == 1:
            x = stim; y = resp
        else:
            x = resp; y = stim
        
        return bs.predict(self,x,y,zeropad = self.Zeropad,**kwargs)
    
    def save(self,path,name):
        output = dict()
        for i in self.__dict__:
            output[i] = self.__dict__[i]
        
        siIO.saveObject(output, path,name, '.mtrf')
        
    def load(self,path):
        temp = siIO.loadObject(path)
        for i in temp:
            setattr(self, i, temp[i])
            
    def cuda(self,debug = False):
        from .CudaCore import CCoreCuda
        oCuda = CCoreCuda()
        Core.oCuda = oCuda
        ds.oCuda = oCuda
        self._oCuda = oCuda
        self._oCuda.DEBUG = debug
        
    def cpu(self):
        Core.oCuda = None
        ds.oCuda = None
        self._oCuda = None
        
    def plotWeights(self,vecNames = None):

        '''desined for models trained with combined vector '''
        from matplotlib import pyplot as plt
        times = self.t
        out = list()
        for i in range(self.w.shape[0]):
            weights = self.w[i,:,:]#take mean along the input dimension
            fig1 = plt.figure()
            plt.plot(times,weights[:,:])
            plt.title(vecNames[i])
            out.append(fig1)
        return out
        

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