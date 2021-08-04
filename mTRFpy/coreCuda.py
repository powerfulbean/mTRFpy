# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:12:48 2021

@author: ShiningStone
"""
import numpy as np
class CCoreCuda:
    
    def __init__(self):
        self.cp = self.getCupy()
        
    def getCupy(self):
        import cupy as cp
        return cp

    def calCovariance(self,x,y):
        '''
        calculat the covariance of two matrices
        x: left matrix
        y: right matrix
        
        if the input for x and y are both 1-D vectors, they will be reshaped to (len(vector),1)
        '''
        if isinstance(x, np.ndarray):
            x = self.cp.asarray(x)
        if isinstance(y, np.ndarray):
            y = self.cp.asarray(y)
        temp = self.cp.matmul(x.T,y)
        self.cp.cuda.Stream.null.synchronize()
        return temp