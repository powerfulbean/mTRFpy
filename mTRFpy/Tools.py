# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:25 2020

@author: Jin Dou
"""
import numpy as np
from . import Core
from . import DataStruct as ds

from . import Protocols as pt
from StellarInfra import IO as siIO
from StellarInfra.DirManage import CDirectoryConfig

loadMatFile = siIO.loadMatFile
saveMatFile = siIO.saveMatFile

oDataPrtcl = pt.CProtocolData()

def cmp2NArray(a,b,digitsNum = None):
    #digitsNum： the number of digits after the decimal point
    if digitsNum != None:
        a = np.around(a,digitsNum)
        b = np.around(b,digitsNum)
    return np.array_equal(a,b)

if __name__ == '__main__':
    
    oDir = CDirectoryConfig(['TestData'],'..\mTRFpy.conf')
    temp = loadMatFile(oDir.TestData + 'speech_data.mat')
    temp1 = saveMatFile( oDir.TestData+ 'speech_data_out.mat',temp)
    
    