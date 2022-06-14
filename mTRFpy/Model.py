# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:40 2020

@author: Jin Dou
"""
import sklearn as skl
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin, TransformerMixin
from sklearn.model_selection import ShuffleSplit,LeaveOneOut
from multiprocessing import Pool,shared_memory
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

IF_MNE = False

try:
    import mne
    IF_MNE = True
except:
    IF_MNE = False


from . import Tools
from . import DataStruct as ds
from . import Basics as bs
from . import Core
import sys
from tqdm import tqdm

DirEnum = tuple([-1,1]) 

def funcTrainNVal(oInput):
    trainIdx = oInput[0]
    testIdx = oInput[1]
    idx = oInput[2]
    nSplits = oInput[3]
    modelParam = shared_memory.ShareableList(name='sharedModelParam') 
    Dir = modelParam[0]
    fs = modelParam[1]
    tmin_ms = modelParam[2]
    tmax_ms = modelParam[3]
    Lambda = modelParam[4]
    stim = modelParam[5]
    resp = modelParam[6]
    oTRF = CTRF()
    stimTrain = stim.selectByIndices(trainIdx)
    respTrain = resp.selectByIndices(trainIdx)
    oTRF.train(stimTrain, respTrain, Dir, fs, tmin_ms, tmax_ms, Lambda)
    
    stimTest = stim.selectByIndices(testIdx)
    respTest = resp.selectByIndices(testIdx)
    _,r,err = oTRF.predict(stimTest,respTest)
    sys.stdout.write("\r" + f"cross validation >>>>>..... split {idx}/{nSplits}\r")
    return (r,err)

def createSharedNArray(a,name):
    shm = shared_memory.SharedMemory(name=name,create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    return b
    

def crossVal(stim:ds.CDataList,resp:ds.CDataList,
             Dir,fs,tmin_ms,tmax_ms,Lambda,
             random_state = 42,nWorkers=1,n_Splits = 10, **kwargs):
    if np.isscalar(Lambda):
        r,err = crossValPerLambda(stim, resp, Dir, fs, tmin_ms, tmax_ms, Lambda,
                                 random_state = random_state,nWorkers=nWorkers,
                                 n_Splits = n_Splits,**kwargs)
        return np.mean(r,axis = 0,keepdims=True),np.mean(err,axis = 0,keepdims=True)
    else:
        result = []
        for l in tqdm(Lambda,desc = 'lambda',leave = False):
            r,err = crossValPerLambda(stim, resp, Dir, fs, tmin_ms, tmax_ms, l, 
                                  random_state = random_state,nWorkers=nWorkers,
                                   n_Splits = n_Splits,**kwargs)
            result.append(
                (np.mean(r,axis = 0,keepdims=True),np.mean(err,axis = 0,keepdims=True))
                )
        return result    
        

def crossValPerLambda(stim:ds.CDataList,resp:ds.CDataList,
             Dir,fs,tmin_ms,tmax_ms,Lambda,
             random_state = 42,nWorkers=1,n_Splits = 10, **kwargs):
    stim = ds.CDataList(stim)
    resp = ds.CDataList(resp)
    if n_Splits is not None:
        nSplits = n_Splits
        testSize = 1/nSplits
        if testSize > 0.1:
            testSize = 0.1
        rs = ShuffleSplit(n_splits = nSplits,test_size=testSize,random_state=random_state)
    else:
        rs = LeaveOneOut()
        nStim = len(stim)
        nSplits = nStim
    
    if nWorkers <= 1:
        finalR = []
        finalErr = []
        idx = 0
        
        for trainIdx,testIdx in tqdm(rs.split(stim),desc = 'cv',leave = False):
            # print('def\rabc')
            # sys.stdout.write(f"cross validation >>>>>..... split {idx+1}/{nStim}")
            # sys.stdout.flush()
            # sys.stdout.write("\033[F") 
            # sys.stdout.write("\033[K")  
            # print("\r" + f"Lambda: {Lambda}; cross validation >>>>>..... split {idx+1}/{nSplits}",end='\r')
            
            idx+=1
            oTRF = CTRF()
            stimTrain = stim.selectByIndices(trainIdx)
            respTrain = resp.selectByIndices(trainIdx)
            oTRF.train(stimTrain, respTrain, Dir, fs, tmin_ms, tmax_ms, Lambda, **kwargs)
            
            stimTest = stim.selectByIndices(testIdx)
            respTest = resp.selectByIndices(testIdx)
            _,r,err = oTRF.predict(stimTest,respTest)
            finalR.append(r)
            finalErr.append(err)
    else:
        splitParam = []
        idx=1
        for trainIdx,testIdx in rs.split(stim):
            splitParam.append([trainIdx,testIdx,idx,nSplits])
            idx+=1
            
        modelParam = shared_memory.ShareableList(
            [Dir,fs, tmin_ms, tmax_ms, Lambda],
            name= 'sharedModelParam')
        
        sharedStim = createSharedNArray(stim, 'sharedStim')
        sharedResp = createSharedNArray(resp, 'sharedResp')
        
        # stop
        with Pool(nWorkers) as pool:
            out = pool.imap(funcTrainNVal, splitParam,chunksize=int(nSplits/nWorkers))
            finalR = [i[0] for i in out]
            finalErr = [i[1] for i in out]
    finalR = np.concatenate(finalR)
    finalErr = np.concatenate(finalErr)
    return finalR,finalErr#np.mean(finalR,axis=0),np.mean(finalErr,axis=0)

class CTRF:
    
    def __init__(self,):
        self.w = None
        self.b = None
        self.t = None
        self.Dir = None
        self.Type = 'multi'
        self.Zeropad = True
        self.fs = -1
        self._oCuda = None
    
    def __radd__(self, oTRF):
        if oTRF == 0:
            return self.copy()
        else:
            return self.__add__(oTRF)
    
    def __add__(self,oTRF):  
        oTRFNew = self.copy()
        oTRFNew.w += oTRF.w
        oTRFNew.b += oTRF.b
        #!!!need check other params
        return oTRFNew
    
    def __truediv__(self,num):
        oTRFNew = self.copy()
        oTRFNew.w /= num
        oTRFNew.b /= num
        return oTRFNew
    
    def train(self,stim,resp,Dir,fs,tmin_ms,tmax_ms,Lambda,**kwargs):
        
        if isinstance(stim, np.ndarray):
            stim = ds.CDataList([stim])
        
        if isinstance(resp, np.ndarray):
            resp = ds.CDataList([resp])
        
        assert Dir in DirEnum
        
        if (Dir == 1):
            x = stim
            y = resp
        else:
            x = resp
            y = stim
            tmin_ms, tmax_ms = Dir * tmax_ms, Dir * tmin_ms
        
        w,b,lags = bs.train(x,y,fs,tmin_ms,tmax_ms,Lambda,oCuda = self._oCuda,**kwargs)
        
        if kwargs.get('Type') != None:
            self.type = kwargs.get('Type')
        
        if kwargs.get('Zeropad') != None:
            self.Zeropad = kwargs.get('Zeropad')
            
        self.w, self.b = w, b
        self.Dir = Dir
        self.t = Core.Idxs2msec(lags,fs)
        self.fs = fs
    
    
    def predict(self,stim,resp = None,**kwargs):
        if isinstance(stim, np.ndarray):
            stim = ds.CDataList([stim])
        
        if isinstance(resp, np.ndarray):
            resp = ds.CDataList([resp])
            
        assert self.Dir in DirEnum
        if self.Dir == 1:
            x = stim; y = resp
        else:
            x = resp; y = stim
        
        return bs.predict(self,x,y,zeropad = self.Zeropad,**kwargs)
    
    def save(self,path,name):
        output = dict()
        for i in self.__dict__:
            output[i] = self.__dict__[i]
        
        Tools.saveObject(output, path,name, '.mtrf')
        
    def load(self,path):
        temp = Tools.loadObject(path)
        for i in temp:
            setattr(self, i, temp[i])
        return self
            
    def copy(self):
        oTRF = CTRF()
        for k,v in self.__dict__.items():
            value = v
            if getattr(v,'copy',None) is not None:
                value = v.copy()
            setattr(oTRF, k, value)
        return oTRF
        
            
    def cuda(self,debug = False):
        from .CudaCore import CCoreCuda
        oCuda = CCoreCuda()
        Core.oCuda = oCuda
        ds.oCuda = oCuda
        self._oCuda = oCuda
        self._oCuda.DEBUG = debug
        
    def cpu(self):
        Core.oCuda = None
        ds.oCuda = None
        self._oCuda = None
        
    def plotWeights(self,vecNames = None,ylim = None,newFig = True,chan = [None]):

        '''desined for models trained with combined vector '''
        from matplotlib import pyplot as plt
        times = self.t
        out = list()
        if self.Dir == 1:
            nStimChan = self.w.shape[0]
        elif self.Dir == -1:
            nStimChan = self.w.shape[-1]
        else:
            raise ValueError
        
        for i in range(nStimChan):
            if self.Dir == 1:
                weights = self.w[i,:,:]#take mean along the input dimension
            elif self.Dir == -1:
                weights = self.w[:,:,i].T
            else:
                raise ValueError
            if newFig:
                fig1 = plt.figure()
            else:
                fig1 = None
            plt.plot(times,weights[:,slice(*chan)])
            plt.title(vecNames[i] if vecNames is not None else '')
            plt.xlabel("time (ms)")
            plt.ylabel("a.u.")
            if ylim:
                plt.ylim(ylim)
            if fig1:
                out.append(fig1)
        return out
    
    def weightsMneObj(self,tarStimChanIdx = 0, chNames = None,montage = None):
        '''

        Parameters
        ----------
        tarStimChanIdx: int
            stimuli's target channel index for exporting the weights    
            The default is 0.
        chNames : list, optional
            list of channel names, The default is None.
        montage: mne.channels.DigMontage, optoinal
            The default is None.
        
        
        Returns
        -------
        The mne object containing the data in self.w

        '''
        if not IF_MNE:
            raise ModuleNotFoundError('the module MNE is not available')
        if self.Dir == -1:
            weights = self.w[:,:,tarStimChanIdx]
        else:
            weights = self.w[tarStimChanIdx].T
        info = mne.create_info(chNames, self.fs,ch_types = 'eeg')
        mneW = mne.EvokedArray(weights,info)
        if montage is not None:
            mneW.set_montage(montage)
        time = np.array(self.t)
        mneW.times = time / 1000
        return mneW
    
    def topoplot(self,tarStimChanIdx, chNames,montage,figpath = None,title = '',**kwargs):
        
        mneW = self.weightsMneObj(tarStimChanIdx, chNames,montage)
        slide = len(self.t) // 20
        if kwargs.get('axes') is not None:
            times = 'auto'
        else:
            times = np.array(self.t) / 1000
            slide = len(times) //20
            times = times[1:-1:slide]
            
        fig = mneW.plot_topomap(times = times,res = 256,sensors=False,
                        cmap='jet',outlines='head',time_unit='s',scalings = 1,
                        title = title + ' weight',units = 'a.u.',cbar_fmt='%3.3f',
                        **kwargs)#,vmin = -3500, vmax = 3500)#
        if figpath:
            fig.savefig(figpath +'/'+ title.replace('.','_') + '_' + 'topoplot')
            plt.close(fig)
        
        return fig
        # slide = len(self.t) // 20
        # fig3 = mneW.plot_topomap(time[::slide+1]/1000,res = 256,sensors=False,
        #                 cmap='jet',outlines='head',time_unit='s',title = '')#,vmin = -3500, vmax = 3500)#
        # temp1 = fig3.get_axes()
        # temp1[0].set_title('')
        # t2 = temp1[-1]
        # t2.set_title('a.u.')
        # t2.set_axis_off()
        # t2.set_visible(False)
        # self._save(fig3, title,'topoplot'.lower())
     
    def _plotWSTD(self,axs,ylim_w,ylim_std):
        if self.Dir == -1:
            w = self.w.T
        else:
            w = self.w
        t = self.t
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[0].set_xticks([])
        if ylim_w is not None:
            axs[0].set_ylim(*ylim_w)  
            ticks = [0,ylim_w[-1]*0.4,ylim_w[-1]]
            axs[0].set_yticks(ticks)
        # ticklabels = axs[0].get_yticklabels()
        axs[0].plot(t,w[0])
        axs[0].tick_params(axis=u'y', which=u'both',direction = 'in')
        axs[1].tick_params(axis=u'both', which=u'both',direction = 'in')
        axs[0].set_ylabel('a.u.')
        # times = [t[0],t[6],t[17]]
        # [axs[0].axvline(x = i,color='black',linewidth=0.8) for i in times]
        
        std = np.std(w[0],axis=1,keepdims = True)
        if ylim_std is not None:
            axs[1].set_ylim(*ylim_std)
            ticks = [0,ylim_std[-1]/2,ylim_std[-1]]
            # print(ticks)
            # ticks = np.round(ticks)#,decimals=3)
            axs[1].set_yticks(ticks)
        ticklabels = axs[1].get_yticklabels()
        ticklabels[-1].set_visible(False)
        # ticks[-2] = max(std)
        axs[1].tick_params(axis=u'both', which=u'both',direction = 'in')
        axs[1].plot(t,std,color='black')
        axs[1].set_ylabel('STD')
        axs[1].set_xlabel('time (ms)')
        # [axs[1].axvline(x = i,color='black',linewidth=0.8) for i in times]
        
     
    def wSTDPlot(self,title="",ylim_w = None,ylim_std =None):
        # model = self.model
        fig = plt.figure(figsize = (20,10))
        gs = fig.add_gridspec(2, hspace=0,height_ratios = [1,0.5])
        axs = gs.subplots(sharex=False)
        self._plotWSTD(axs,ylim_w,ylim_std)
        return fig
    
    # def plotWSTDAndTopoplot(self,tarStimChanIdx,chanloc,title='',kwargsTopo = {}):
    #     fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    #     subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[2, 1])
    #     gs1 = GridSpec(2, 1, figure=subfigs[0])
    #     ax0 = subfigs[0].add_subplot(gs1[0, :])
    #     ax1 = subfigs[0].add_subplot(gs1[1, :])
    #     gs2 = GridSpec(2, 10, figure=subfigs[1])
    #     axs1 = [subfigs[1].add_subplot(gs2[0, i]) for i in range(10)]
    #     axs2 = [subfigs[1].add_subplot(gs2[1, i]) for i in range(10)]
    #     axs = axs1 + axs2
    #     self.topoplot(tarStimChanIdx,chanloc[0],chanloc[1],axes = axs,**kwargsTopo)
    #     return fig
        
    
    def plotWSTDAndTopoplot(self,tarStimChanIdx,chanloc,title='',
                            ylim_w = None,ylim_std = None,kwargsTopo = {}):
        
        fig = plt.figure()
        gs = GridSpec(4, 11, figure=fig)
        axs1 = [fig.add_subplot(gs[2, i]) for i in range(10)]
        axs2 = [fig.add_subplot(gs[3, i]) for i in range(10)]
        axs = axs1 + axs2
        kwargsTopo['vmin'] = ylim_w[0] if ylim_w else None
        kwargsTopo['vmax'] = ylim_w[1] if ylim_w else None
        fig = self.topoplot(tarStimChanIdx,chanloc[0],chanloc[1],axes = axs,title=title,**kwargsTopo)
        ax0 = fig.add_subplot(gs[0, 1:-1])
        ax1 = fig.add_subplot(gs[1, 1:-1])
        self._plotWSTD([ax0,ax1],ylim_w = ylim_w,ylim_std = ylim_std)#, ylim_w = [-0.1,1],ylim_std = [0,1])
        return fig
        
        

class CSKlearnTRF(BaseEstimator,RegressorMixin, TransformerMixin, CTRF):
    '''
    main difference is that Dir will always be 1
    
    '''
    
    def __init__(self,fs, tmin_ms, tmax_ms, Lambda,**kwargs):
        super().__init__()
        self.Dir = 1
        self.fs = fs
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.Lambda = Lambda
        self.Type = 'multi'
        self.Zeropad = True
        self.kwargs = kwargs
        
    def fit(self,x,y):
        x = skl.utils.check_array(x)
        y = skl.utils.check_array(y)
        self.train(x,y,self.Dir,self.fs,self.tmin_ms,self.tmax_ms,self.Lambda,**self.kwargs)
    
    def predict(self,x):
        pass
    
    def transform(self,x):
        pass