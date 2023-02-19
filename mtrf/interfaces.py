# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:29 2023

@author: powerfulbean
"""
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

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
            self.mne_info = mne.create_info(ch_names,sfreq,
                                            ch_types = ch_types,montage = montage)
        self.montage = montage
        self.ch_names = ch_names
        
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
    
    def plot_topo(self,data,title = '',ax = None,chanIdxToMark = None,units = 'a.u.',**kwargs):
        info = self.mne_info        
        times = 'auto'
        kwargs['sensors'] = False if 'sensors' not in kwargs else kwargs['sensors']
        kwargs['res'] = 256 if 'res' not in kwargs else kwargs['res']
        chNames = self.mne_info['ch_names']
        fs = self.mne_info['sfreq']
        info = mne.create_info(chNames, fs,montage = self.montage,ch_types = 'eeg')
        mask = None
        if chanIdxToMark is not None:
            mask = np.zeros(data.shape,dtype = bool)
            for i in chanIdxToMark:
                mask[i] = True
        maskParam = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
            linewidth=0, markersize=8)
        
        if data.ndim == 2:
            evokedArray = self.to_EvokedArray(data)
            fig = evokedArray.plot_topomap(times = times,
                                cmap='jet',outlines='skirt',time_unit='s',
                                scalings = 1,
                                title = title,units = units,cbar_fmt='%3.3f',
                                mask = mask,
                                mask_params= maskParam,
                                **kwargs)#,vmin = -3500, vmax = 3500)#
        elif data.ndim == 1:
            ifAx = False
            if 'ax' in kwargs:
                ax1 = kwargs['ax']
                del kwargs['ax']
                ifAx = True
            else:
                fig = plt.figure(tight_layout=True)
                gridspec_kw={'width_ratios': [19, 1]}
                gs = gridspec.GridSpec(4, 2,**gridspec_kw)
                ax1 = fig.add_subplot(gs[:, 0])
                ax2 = fig.add_subplot(gs[1:3, 1])
    
            maskParam2 = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=4)
            im,cm = mne.viz.plot_topomap(data.squeeze(),info,cmap='jet',
                                outlines='skirt',
                                axes = ax1,
                                mask = mask,names = chNames,
                                mask_params= maskParam2,
                                **kwargs)
            # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
            if not ifAx:
                clb = fig.colorbar(im, cax=ax2)
                clb.ax.set_title(units,fontsize=10) # title on top of colorbar
                fig.suptitle(title)
        else:
            raise NotImplementedError
        
        if not ifAx:
            return fig
        else:
            return im
            