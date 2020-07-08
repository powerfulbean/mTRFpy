# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:24:23 2020

@author: Jin Dou
"""
from abc import ABC, abstractmethod
import numpy as np

class CProtocol:
    
    def __init__(self):
        pass
    
    def __call__(self,*args, **kwargs):
        return self.protocol(*args,**kwargs)
      
    @abstractmethod
    def protocol(self,param):
        pass
    
class CProtocolData(CProtocol):
    
    def __init__(self):
        super().__init__()
        
    def protocol(self,*dataList):
        #verify that the the rows correspond to samples and the columns to variables
        output = list()
        for data in dataList:
            data = np.array(data)
            
            #if the input for x and y are both 1-D vectors, they will be reshaped to (len(vector),1)
            if len(data.shape) == 1:
                data = np.expand_dims(data,1)
            
            #perform other checks
            if False:
                raise ValueError()
                
            output.append(data)
            
        #end of for
        
#        if len(output) == 0:
#            if 
#            return None
#        else:
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)