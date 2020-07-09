# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:32:05 2020

@author: Jin Dou
"""
from scipy import io as scipyIO
from StellarInfra.DirManage import CDirectoryConfig

def loadMatFile(matFilePath):
    return scipyIO.loadmat(matFilePath)

def saveMatFile(matFilePath,mdict,**kwargs):
    return scipyIO.savemat(matFilePath,mdict,**kwargs)


if __name__ == '__main__':
    
    oDir = CDirectoryConfig(['TestData'],'..\mTRFpy.conf')
    temp = loadMatFile(oDir.TestData + 'speech_data.mat')
    temp1 = saveMatFile( oDir.TestData+ 'speech_data_out.mat',temp)
    
    