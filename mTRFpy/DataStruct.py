# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:50:12 2020

@author: Jin Dou
"""
from collections.abc import Iterable
from . import Protocols as pt
import numpy as np
class CDataList(list):
    
    def __init__(self,data,dim:int = 0,split:int = 1):
        list().__init__([])
        self.dim = dim
        self.split = split
        data = self._check(data)
        data = self._split(data)
        self.extend(data)
        
    def _check(self,data):
        if not isinstance(data,list):
            data = [data]
        data = pt.CProtocolData()(*data)
        nColSizeType = len(set([d.shape[1] for d in data]))
        assert nColSizeType == 1 #arrays in the list should have the same size
        return data
    
    def _split(self,data:list):
        output = list()
        for d in data:
            lenSeg = int(np.ceil(len(d)/ self.split))
            for i in range(self.split):
                rowSlice = slice(i * lenSeg,(i+1) * lenSeg)
                output.append(d[rowSlice])
        return output
    
    @property
    def fold(self):
        return self.__len__()
    
    @property
    def nVar(self):
        array = self.__getitem__(0)
        return array.shape[1]
    
    
def DataListOp(funcOp):
    def wrapper(*args, **kwargs):
        oDataListArgs = list() #a list of CDataList
        otherArgs = list()
        for arg in args:
            #extract the CDataList type arguments
            if isinstance(args,CDataList):
                oDataListArgs.append(arg)
            else:
                otherArgs.append(arg)
        
        if len(oDataListArgs) == 0:
            raise ValueError('at least one CDataList should be provided' )
            
        output = list()
        for idx in range(oDataListArgs[0].fold):
            #prepare the 'idx'th data in CDataList
            curDataArg = [oDataList[idx] for oDataList in oDataListArgs]
            curArgs = curDataArg + otherArgs
            output.append(funcOp(*curArgs,**kwargs))
        return output
    return wrapper
