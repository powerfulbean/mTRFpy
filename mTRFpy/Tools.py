# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:25 2020

@author: Jin Dou
"""
import os
import warnings
import numpy as np
from scipy import io as scipyIO

from StimRespFlow.outsideLibInterfaces import CIfMNE
from matplotlib import pyplot as plt


class CDTRFWeights:
    '''
    Used to visualize the weights of TRF, and other anaylsis
    '''
    def __init__(self,w,t,model,srate,chanlocs,figpath = None):
        self._w = w
        self._t = t
        self._data = model
        chNames,montage = chanlocs
        self.oMNE = CIfMNE(chNames,srate,'eeg',montage)
        self.figpath = figpath
    
    @property
    def w(self):
        return self._w
    
    @property
    def t(self):
        return self._t
    
    @property
    def model(self):
        return self._data
    
    def plotSTDTRF(self,title=""):
        # model = self.model
        w = self.w
        t = self.t
        fig = plt.figure(figsize = (20,10))
        gs = fig.add_gridspec(2, hspace=0,height_ratios = [1,0.5])
        axs = gs.subplots(sharex=False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[0].set_ylim(-1,1)
        ticks = axs[0].get_yticks()
        axs[0].set_xticks([])
        ticks[-1] = 1
        axs[0].set_yticks(ticks[[0,2,-1]])
        ticklabels = axs[0].get_yticklabels()
        axs[0].plot(t,w[0])
        axs[0].tick_params(axis=u'y', which=u'both',direction = 'in')
        axs[1].tick_params(axis=u'both', which=u'both',direction = 'in')
        axs[0].set_ylabel('a.u.')
        times = [t[0],t[6],t[17]]
        [axs[0].axvline(x = i,color='black',linewidth=0.8) for i in times]
        
        std = np.std(w[0],axis=1,keepdims = True)
        axs[1].set_ylim(0,0.1)
        ticklabels = axs[1].get_yticklabels()
        ticklabels[-1].set_visible(False)
        ticks = axs[1].get_yticks()
        ticks = np.arange(0,0.2+0.05,0.05)
        # ticks[-2] = max(std)
        ticks = np.round(ticks,decimals=3)
        axs[1].set_yticks(ticks)
        axs[1].tick_params(axis=u'both', which=u'both',direction = 'in')
        axs[1].plot(t,std,color='black')
        axs[1].set_ylabel('STD')
        axs[1].set_xlabel('time (ms)')
        [axs[1].axvline(x = i,color='black',linewidth=0.8) for i in times]
        plt.title(title)
        self._save(fig, title,'STDTRF'.lower())
    
    @property
    def weights(self):
        return self._w
    
    def plotTopomap(self,inChanIdx,timeIdx=None,title= ''):
        if self.model.Dir == -1:
            oWeight = self.oMNE.getMNEEvoked(self.weights[:,:,inChanIdx])
        else:
            oWeight = self.oMNE.getMNEEvoked(self.weights[inChanIdx].T)
        time = np.array(self.t)
        oWeight.times = time / 1000
        slide = len(self.t) // 20
        fig3 = oWeight.plot_topomap(time[::slide+1]/1000,res = 256,sensors=False,
                        cmap='jet',outlines='head',time_unit='s',title = title)#,vmin = -3500, vmax = 3500)#
        temp1 = fig3.get_axes()
        temp1[0].set_title('')
        t2 = temp1[-1]
        t2.set_title('a.u.')
        # t2.set_axis_off()
        # t2.set_visible(False)
        self._save(fig3, title,'topoplot'.lower())
        return oWeight
    
    def _save(self,figHandle,title,tag):
        if self.figpath:
            figHandle.savefig(self.figpath +'/'+ title.replace('.','_') + '_' + tag)
            plt.close(figHandle)

def cmp2NArray(a,b,digitsNum = None):
    #digitsNum： the number of digits after the decimal point
    if digitsNum != None:
        a = np.around(a,digitsNum)
        b = np.around(b,digitsNum)
    return np.array_equal(a,b)

def checkFolder(folderPath):
#    print(folderPath)
#    if not isinstance(folderPath,str):
#        return
    if not os.path.isdir(folderPath) and not os.path.isfile(folderPath):
        warnings.warn("path: " + folderPath + " doesn't exist, and it is created")
        os.makedirs(folderPath)

''' Python Object IO'''
def saveObject(Object,folderName,tag=None, ext = '.bin'):
    if tag is None:
        file = open(folderName, 'wb')
    else:
        checkFolder(folderName)
        file = open(folderName + '/' + str(tag) + ext, 'wb')
    import pickle
    pickle.dump(Object,file)
    file.close()
    
def loadObject(filePath):
    import pickle
    file = open(filePath, 'rb')
    temp = pickle.load(file)
    return temp
''' End '''



'''load Matlab .Mat file '''

def loadMatFile(matFilePath):
    return scipyIO.loadmat(matFilePath)

def saveMatFile(matFilePath,mdict,**kwargs):
    return scipyIO.savemat(matFilePath,mdict,**kwargs)
''' End '''


class StageControl:
    
    def __init__(self,tartgetList):
        self.targetList = tartgetList
        
    def stage(self,stageNum):
        if(stageNum in self.targetList):
            print('Stage_' + str(stageNum) + ':')
            return True
        else:
            return False
        
    def __call__(self,stageNum):
        return self.stage(stageNum)
    