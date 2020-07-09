# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:44:37 2020

@author: Jin Dou
"""

from StellarInfra.DirManage import CDirectoryConfig
from mTRFpy import InterfaceMatlab as ifMatlab
from mTRFpy import Operations as op

oDir = CDirectoryConfig(['TestData'],'mTRFpy.conf')
temp = ifMatlab.loadMatFile(oDir.TestData + 'speech_data.mat')











#temp1 = ifMatlab.saveMatFile( oDir.TestData+ 'speech_data_out.mat',temp)
