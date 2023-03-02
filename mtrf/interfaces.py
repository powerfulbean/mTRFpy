# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:29 2023

@author: powerfulbean
"""
import logging
from packaging import version

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


def _check_mne_info(mne_info = None,ch_names = None,sfreq = None,
                 ch_types = None,montage = None):
    if mne is None:
        raise ValueError('mne library is not available')
    
    if mne_info is not None:
        return mne_info
    elif not((sfreq is None) or (ch_types is None)):
        if montage is None:
            montage = mne.channels.make_standard_montage('biosemi128')
        if ch_names is None:
            ch_names = montage.ch_names
        if version.parse(mne.__version__) < version.parse("0.20.0"):
            return mne.create_info(ch_names = ch_names,sfreq = sfreq,
                                   ch_types = ch_types,montage = montage)  
        else:
            mne_info = mne.create_info(ch_names = ch_names,sfreq = sfreq,
                                   ch_types = ch_types)
            mne_info.set_montage(montage)
            return mne_info
    else:
        raise ValueError('not enough information provided for creating a mne array')
                            
    
def to_mne_evoked(data,mne_info = None, ch_names = None, sfreq = None,
                      ch_types = None, montage = None,**kwargs):
    """
    generate a mne.EvokedArray
    Arguments:
        data (np.ndarray | mtrf.TRF): Either a 2-D samples-by-features, 
            (nSamples, nFeatures | nChannels), or a mtrf.TRF object
        times (None | list): Either a list of timestamps, or infered by
            mne.EvokedArray
        kwargs: other arguments that the user want to feed to the 
            mne.EvokedArray
    Returns:
        single or a list of the constructed evokedArray
    """
    mne_info = _check_mne_info(mne_info,ch_names,sfreq,ch_types,montage)
    
    if isinstance(data, TRF):
        w = [mne.EvokedArray(w.T,mne_info,tmin = data.times[0],**kwargs) for w in data.ftc_weights]
        return w
    elif isinstance(data,np.ndarray):
        return mne.EvokedArray(data.T,mne_info,**kwargs)
    else:
        data = np.asarray(data)
        return mne.EvokedArray(data.T,mne_info,**kwargs)


def kwargs_trf_mne_joint(trf = None):
    kwargs = {}
    kwargs['topomap_args'] = {'scalings':1}
    kwargs['ts_args'] = {'units':'a.u.','scalings':dict(eeg=1)}
    return kwargs    

def kwargs_trf_mne_topo(trf = None):
    kwargs = { 'cmap':'jet','outlines':'skirt','time_unit':'s',
        'scalings' : 1, 'units' : 'a.u.','cbar_fmt':'%3.3f'}
    return kwargs

def kwargs_r_mne_topo(trf = None):
    kwargs = {'times':[0], 'cmap':'jet','time_unit':'s',
        'scalings' : 1, 'units' : 'r','cbar_fmt':'%3.3f'}
    return kwargs