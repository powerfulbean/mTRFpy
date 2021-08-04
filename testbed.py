import numpy as np

from mTRFpy import Operations as op
from mTRFpy import DataStruct as ds
from mTRFpy import Model as md
from StellarInfra.StageControl import CStageControl

oStage = CStageControl([6])

if oStage(1):
    a = np.array([[0,1,2,3,4,5,6]])
    b = np.concatenate([a,a],0)
    c = b.T.copy()
    
    test1 = op.calCovariance(a,a)
    test2 = op.calCovariance([1,2,3],[1,2,3])
    test3 = op.calCovariance(b,b)
    
    test4 = op.genLagMat(c,[-3,-1,0,1,3])
    test5 = op.genLagMat([1,2,3,4,5],[-3,-1,0,1,3])
    test6 = op.genLagMat(np.array([1,2,3,4,5]),[-3,-1,0,1,3])
    test7 = op.genLagMat(np.array([[1,2,3,4,5]]),[-3,-1,0,1,3])

if oStage(2):
    test8 = op.genSmplIdxSeqByMsecRg([-100,400],128)
    
    
if oStage(3):
    temp = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
    oList = ds.CDataList(temp,2,2)
    temp1 = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    oList1 = ds.CDataList(temp1)
    temp2 = np.array([[1,2],[1,2],[1,2],[1,2]])
    oList2 = ds.CDataList([temp, temp2])
    
if oStage(4):
    oMd = md.CTRF()
    
if oStage(5):
    op.truncate([1,2,3],-5,5)
    
if oStage(6):
    '''
    test cupy
    '''
    import cupy as cp
    import numpy as np
    import time
    s = time.time()
    x_cpu = np.ones((10,10,10))
    e = time.time()
    print(e - s)
    ### CuPy and GPU
    s = time.time()
    x_gpu = cp.ones((10,10,10))
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print(e - s)
    