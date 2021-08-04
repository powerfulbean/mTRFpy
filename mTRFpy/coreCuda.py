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
        self.memPool = self.cp.get_default_memory_pool()#self.cp.cuda.MemoryPool(self.cp.cuda.malloc_managed)
        # self.cp.cuda.set_allocator(self.memPool.malloc)
        # mempool = self.cp.get_default_memory_pool()
        # mempool.set_limit(size=10.5*1024**3)
        
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
            mempool = self.memPool
            pinned_mempool = self.cp.get_default_pinned_memory_pool()
            print(mempool.get_limit())
            print(mempool.used_bytes())              # 0
            print(mempool.total_bytes())             # 0
            print(pinned_mempool.n_free_blocks())    # 0
        temp = self.cp.matmul(x.T,y)
        self.cp.cuda.Stream.null.synchronize()
        if self.cp.cuda.runtime.getDeviceCount() == 1:
            out = self.cp.asnumpy(temp)
        else:
            with self.cp.cuda.Device(1):
                out = self.cp.array(temp)
        del x
        del y
        del temp
        self.memPool.free_all_blocks()
        return out
    
    def calSelfCovariance(self,x):
        '''
        calculat the covariance of two matrices
        x: left matrix
        y: right matrix
        
        if the input for x and y are both 1-D vectors, they will be reshaped to (len(vector),1)
        '''
        if isinstance(x, np.ndarray):
            x = self.cp.asarray(x)
        if self.DEBUG:
            mempool = self.memPool
            pinned_mempool = self.cp.get_default_pinned_memory_pool()
            print(mempool.get_limit())
            print(mempool.used_bytes())              # 0
            print(mempool.total_bytes())             # 0
            print(pinned_mempool.n_free_blocks())    # 0
        temp = self.cp.matmul(x.T,x)
        self.cp.cuda.Stream.null.synchronize()
        out = self.cp.asnumpy(temp)
        del x
        del temp
        self.memPool.free_all_blocks()
        return out