# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:29 2023

@author: powerfulbean
"""
import logging
import numpy as np
from .model import TRF

try:
    import mne
except:
    logging.warn('mne is not installed, \
                 the functionality based on mne is not avaiable')
    mne = None

class MNE:
    
    def __init__(self,ch_names = None,sfreq = None,
                 ch_types = None,montage = None):
        self.mne_info = None
        if mne is None:
            raise ValueError('mne library is not available')
        if not((ch_names is None) or (sfreq is None) or (ch_types is None)):
            self.mne_info = mne.create_info(ch_names,sfreq,ch_types)
        self.montage = montage
        
    def to_EvokedArray(self,data,times = None,**kwargs):
        """
        generate a mne.EvokedArray
        Arguments:
            data (np.ndarray | mtrf.TRF): Either a 2-D samples-by-features, 
                (nSamples, nFeatures | nChannels)
            times (None | list): Either a list of timestamps, or infered by
                mne.EvokedArray
            kwargs: other arguments that the user want to feed to the 
                mne.EvokedArray
        Returns:
            single or a list of the constructed evokedArray
        """
        
        def _to_EvokedArray(data,times):
            evokedArray = mne.EvokedArray(data.T,self.mne_info,**kwargs)
            if times is not None:
                evokedArray.times = times
            if self.montage is not None:
                evokedArray.set_montage(self.montage)
            return evokedArray
        
        if isinstance(data, TRF):
            w = [_to_EvokedArray(w,data.times) for w in data.ftc_weights]
            b = [_to_EvokedArray(data.bias,times)]
            return w + b
        elif isinstance(data,np.ndarray):
            return _to_EvokedArray(data,times)
        else:
            raise NotImplementedError()
    
    def plot_joint(self,obj,featIdx = None,title = ''):
        #if the data to plot is 3D
        #the first dimension must be for feature
        figs = []
        data = None
        #if the data to plot is 3D
        #the first dimension must be for feature
        if isinstance(obj, TRF):
            obj:TRF = obj
            data = obj.ftc_weights
        elif isinstance(obj, np.ndarray):
            data:np.ndarray = obj
            assert data.ndim >= 2
            if data.ndim == 2:
                data = np.expand_dims(data,0)
        else:
            raise NotImplementedError()
        
        if featIdx is None:
            featIdx = list(range(data.shape[0]))
        elif isinstance(featIdx,int):
            featIdx = [featIdx]
        else:
            featIdx = list(featIdx)
            
        for fIdx in featIdx:
            evokedArr = self.to_EvokedArray(data[fIdx],obj.times)
            fig = evokedArr.plot_joint(
                title = title,
                topomap_args = {'scalings':1}, 
                ts_args = {'units':'a.u.','scalings':dict(eeg=1)})
            fig.axes[0].set_ylabel('a.u.')
            figs.append(fig)
                
        return figs if len(figs) != 1 else figs[0]
            