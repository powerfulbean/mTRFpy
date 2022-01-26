# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:50:12 2020

@author: Jin Dou
"""
from collections.abc import Iterable
from . import Protocols as pt
import numpy as np
from scipy.stats import zscore

# import sys
from memory_profiler import profile
oCuda = None
def cmp2NArray(a,b,decimalNum = None):
    if decimalNum != None:
        a = np.around(a,decimalNum)
        b = np.around(b,decimalNum)
    return np.array_equal(a,b)

class CDataList(list):
    
    def __init__(self,data=None,dim:int = 0,split:int = 1):
        list().__init__([])
        self.oPrtcl = pt.CProtocolData()
        self.dim = dim
        self.split = split
        if data is None:
            return
        # data = self._check(data)
        # data = self._split(data)
        self.extend(data)
        
    def _check(self,data):
        if not isinstance(data,list):
            data = [data]
        data = self.oPrtcl(*data) #copy the data items in this function
        nColSizeType = len(set([d.shape[1] for d in data]))
        assert nColSizeType == 1 #arrays in the list should have the same nVar
        return data
    
    def _split(self,data:list):
        output = list()
        for d in data:
            lenSeg = int(np.ceil(len(d)/ self.split))
            for i in range(self.split):
                rowSlice = slice(i * lenSeg,(i+1) * lenSeg)
                output.append(d[rowSlice])
        return output
    
    def __add__(self,Input):
        assert isinstance(Input,CDataList)
        return CDataList(list(self)+list(Input))
    
    def copy(self):
        return CDataList(self,self.dim,self.split)
    
    def append(self,data):
        # data = self.oPrtcl(*[data])
        super().append(data)
        
    def equals(self,dataList,decimalNum = None):
        return all([cmp2NArray(self.__getitem__(idx),i,decimalNum) for idx,i in enumerate(dataList)])
    
    @property
    def fold(self):
        return self.__len__()
    
    @property
    def nVar(self):
        array = self.__getitem__(0)
        return array.shape[1]
    
    @property
    def T(self):
        return CDataList([i.T for i in self])
    
    def getChan(self,chanId):
        return CDataList([i[:,chanId:chanId+1] for i in self])
    
    def zscored(self,axis):
        out = list()
        for i in self:
            out.append(zscore(i,axis))
        return CDataList(out)
            
    
class CDataset:
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
class CDatasetDiskSave(CDataset):    
    
    def __init__(self,dataset,indicesConfig,x = True,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dataset = dataset
        self.indicesConfig = indicesConfig
        self.x = x
        self._nVar = self._getNVar()
        
    def __getitem__(self, idx):
        realIdx = self.indicesConfig[idx]
        resp = self.dataset[realIdx].data.T
        stim = self.dataset.stimuliDict[self.dataset[realIdx].stimuli['wordVecKey']][:,0:self.dataset[realIdx].stimuli['sharedLen']].T 
        # self.dataset.clearRef([realIdx])
        return stim,resp
    
    def get(self, idx):
        stim,resp = self.__getitem__(idx)
        realIdx = self.indicesConfig[idx]
        self.dataset.clearRef([realIdx])
        return stim,resp
    
    @property
    def fold(self):
        return len(self.indicesConfig)
        
    @property
    def nVar(self):
        return self._nVar
        
    def _getNVar(self):
        temp = self.__getitem__(0)
        if self.x:
            return temp[0].shape[1]
        else:
            return temp[1].shape[1]
        
    def sparse(self):
        from scipy.sparse import csr_matrix
        for i in self.dataset.stimuliDict:
            self.dataset.stimuliDict[i] = csr_matrix(self.dataset.stimuliDict[i])

def findSecondNonZeroIdx(data):
    data = data.copy()
    t1 = np.argmax(data>0,axis = 1)
    # print(t1)
    for idx,i in enumerate(t1):
        data[idx,i] = 0
    # print(data)
    t1 = np.argmax(data>0,axis = 1)
    return t1[0]
    

# @profile
def buildDataset(dataset,indicesConfig):
    # print(type(dataset),indicesConfig)
    
    # resp = [dataset[i].data.T[2*64:,:] for i in indicesConfig]
    # stim = [dataset.stimuliDict[dataset[i].stimuli['wordVecKey']][:,0:dataset[i].stimuli['sharedLen']].T[64* 2:,:] for i in indicesConfig]
    resp = list()
    stim = list()
    for i in indicesConfig:
        stimTemp = dataset.stimuliDict[dataset[i].stimuli['wordVecKey']][:,0:dataset[i].stimuli['sharedLen']]
        dataset[i].data = dataset[i].data[:,0:dataset[i].stimuli['sharedLen']]
        tarIdx = findSecondNonZeroIdx(stimTemp)
        resp.append(dataset[i].data.T[tarIdx:,:])
        stim.append(stimTemp.T[tarIdx:,:])
    # dataset.clearRef(indicesConfig)
    resp = CDataList(resp)
    stim = CDataList(stim)
    resp = resp.zscored(0)
    stim = stim.zscored(0)
    return stim,resp

def buildResidualDataset(model,dataset,indicesConfig,idxForRes):
    stim,resp = buildDataset(dataset, indicesConfig)
    stimRes = [i[:,1:] for i in stim]
    stimOut = [i[:,0:1] for i in stim]
    # stimRes = [i[:,0:1] for i in stim]
    # stimOut = [i[:,1:] for i in stim]
    stimRes = CDataList(stimRes)
    stimOut = CDataList(stimOut)
    respRes = model.predict(stimRes)
    respOut = [resp[idx] - respRes[idx] for idx,i in enumerate(respRes)]
    respOut = CDataList(respOut)
    return stimOut,respOut
    
def DataListOp(funcOp):
    # @profile
    def wrapper(*args, **kwargs):
        oDataListArgs = list() #a list of CDataList
        otherArgs = list()
        for arg in args:
            #extract the CDataList type arguments
            if isinstance(arg,CDataList) or isinstance(arg,CDataset):
                oDataListArgs.append(arg)
            else:
                otherArgs.append(arg)
        
        if len(oDataListArgs) == 0:
            raise ValueError('at least one CDataList should be provided' )
            
        output = list()
        # print(type(oDataListArgs[0]))
        if isinstance(oDataListArgs[0],CDataList):
            for idx in range(oDataListArgs[0].fold):
                #prepare the 'idx'th data in CDataList
                print('\rfold: ',idx,end='\r')
                curDataArg = [oDataList[idx] for oDataList in oDataListArgs]
                curArgs = curDataArg + otherArgs
                # output.append(funcOp(*curArgs,**kwargs))
                temp = funcOp(*curArgs,**kwargs)
                if len(output) == 0:
                    for i in temp:
                        output.append(i)
                else:
                    for idx,i in enumerate(temp):
                        if oCuda:
                            if oCuda.cp.cuda.runtime.getDeviceCount() == 1:
                                output[idx] = output[idx] + i
                            else:
                                with oCuda.cp.cuda.Device(1):
                                    output[idx] = output[idx] + i
                        else:
                            output[idx] = output[idx] + i
                del temp
            return output
        elif isinstance(oDataListArgs[0],CDataset):
            dataset = oDataListArgs[0]
            for idx in range(oDataListArgs[0].fold):
                print('\rfold: ',idx,end='\r')
                #prepare the 'idx'th data in CDataList
                curDataArg = dataset.get(idx)
                curArgs = list(curDataArg) + otherArgs
                # print(curArgs)
                temp = funcOp(*curArgs,**kwargs)
                if len(output) == 0:
                    for i in temp:
                        output.append(i)
                else:
                    for idx,i in enumerate(temp):
                        if oCuda:
                            if oCuda.cp.cuda.runtime.getDeviceCount() == 1:
                                output[idx] = output[idx] + i
                            else:
                                with oCuda.cp.cuda.Device(1):
                                    output[idx] = output[idx] + i
                        else:
                            output[idx] = output[idx] + i
                del temp
                if oCuda:
                    oCuda.memPool.free_all_blocks()
            return output
        else:
            raise ValueError
        # print('\n')
    return wrapper
