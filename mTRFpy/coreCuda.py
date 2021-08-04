# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:12:48 2021

@author: ShiningStone
"""
import numpy as np
class CCoreCuda:
    
    def __init__(self):
        self.cp = self.getCupy()
        self.DEBUG = False
        self.cp.cuda.set_allocator(self.cp.cuda.MemoryPool(self.cp.cuda.malloc_managed).malloc)
        mempool = self.cp.get_default_memory_pool()
        mempool.set_limit(size=10.5*1024**3)
        
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
        if self.DEBUG:
            mempool = self.cp.get_default_memory_pool()
            pinned_mempool = self.cp.get_default_pinned_memory_pool()
            print(mempool.get_limit())
            print(mempool.used_bytes())              # 0
            print(mempool.total_bytes())             # 0
            print(pinned_mempool.n_free_blocks())    # 0
        temp = self.cp.matmul(x.T,y)
        self.cp.cuda.Stream.null.synchronize()
        return temp