import numpy as np

from mTRFpy import Operations as op

a = np.array([[0,1,2,3,4,5,6]])
b = np.concatenate([a,a],0)
c = b.T.copy()

test1 = op.calCovariance(a,a)
test2 = op.calCovariance([1,2,3],[1,2,3])
test3 = op.calCovariance(b,b)

test4 = op.genLagMatrix(c,[-3,-1,0,1,3])
test5 = op.genLagMatrix([1,2,3,4,5],[-3,-1,0,1,3])
test6 = op.genLagMatrix(np.array([1,2,3,4,5]),[-3,-1,0,1,3])
test7 = op.genLagMatrix(np.array([[1,2,3,4,5]]),[-3,-1,0,1,3])