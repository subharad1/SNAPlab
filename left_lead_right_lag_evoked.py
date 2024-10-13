#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:53:28 2024

@author: sharadhib
"""

import mne
from anlffr.helper import biosemi2mne as bs
import sys
import warnings
from anlffr.preproc import find_blinks
from matplotlib import pyplot as plt
from mne import compute_proj_epochs
from mne import concatenate_epochs
import pylab as pl
import numpy as np
import os
import fnmatch
from mne.time_frequency import tfr_multitaper
from scipy.fft import fft
from scipy.signal import find_peaks
import alpha_spec_functions 

# %%
fpath = "/Users/sharadhib/Library/CloudStorage/Box-Box/SNAPlab/Data/"
subjects = ['SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP008' , 'HR']
resultsdir = 'results/'

for subj in subjects:
    print(subj)
    leads = []
    lags = []
    
    channels_to_drop = ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
    
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_Att*.bdf')
    print(bdfs)
    
    for k, bdf in enumerate(bdfs):
        print(bdf)
        raw, eves = bs.importbdf(bdf, hptsname='biosemi10_20.hpts')
        print(f'Running file: {bdf}')
        raw.notch_filter(np.arange(60, 241, 60), filter_length='auto', fir_design='firwin')
        raw.filter(1, 20)
    
        if subj == 'SP004':
            raw.drop_channels(channels_to_drop)
    
        blinks = find_blinks(raw, ch_name=['Fp1'])
        if subj == 'HR' or subj == 'SP004':
            if k == 0:
                epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),
                                           reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
                blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=2)
            
            raw.add_proj(blink_proj)
            #raw.del_proj(1)
            raw.apply_proj()
        
        else:
            if k == 0:
                epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),
                                           reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
                blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=3)
            
            raw.add_proj(blink_proj)
            raw.del_proj(1)
            raw.apply_proj()
    
        events_lead = [7,9]
        events_lead_use = [event_lead for event_lead in events_lead if event_lead in eves[:,2]]
        
        events_lag = [8,10]
        events_lag_use = [event_lag for event_lag in events_lag if event_lag in eves[:,2]]
    
        if events_lead_use:
            epochs_lead = mne.Epochs(raw, eves, event_id=events_lead_use, baseline=(-0.5, 0.0), proj=True,
                            tmin=-0.5, tmax=5, reject=dict(eeg=200e-6))
        if events_lag_use:
            epochs_lag = mne.Epochs(raw, eves, event_id=events_lag_use, baseline=(-0.5, 0.0), proj=True,
                            tmin=-0.5, tmax=5, reject=dict(eeg=200e-6))
        leads += [epochs_lead, ]
        lags += [epochs_lag, ]
    
    lead = concatenate_epochs(leads)
    lag = concatenate_epochs(lags)    
    

    evoked_lead = lead.average()
    evoked_lag = lag.average()
    
    fname_left = fpath+resultsdir+subj + '_evoked_lead_ave.fif'
    fname_right = fpath+resultsdir+subj + '_evoked_lag_ave.fif'
    
    evoked_lead.save(fname_left,overwrite=True)
    evoked_lag.save(fname_right, overwrite=True)
    
# %%
# =============================================================================
# fpath = '/Users/sharadhib/Library/CloudStorage/Box-Box/SNAPlab/Data/results/'
# subjects = ['SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP008' , 'HR']
# =============================================================================

evoked_leads = [f'{subject_id}_evoked_lead_ave.fif' for subject_id in subjects]
evoked_lags = [f'{subject_id}_evoked_lag_ave.fif' for subject_id in subjects]

evoked_lead = [mne.read_evokeds(leads)[0] for leads in evoked_leads]
evoked_lag = [mne.read_evokeds(lags)[0] for lags in evoked_lags]

average_evoked_lead = mne.grand_average(evoked_lead)
average_evoked_lag = mne.grand_average(evoked_lag)

x = average_evoked_lead.data*1e6
y = average_evoked_lag.data*1e6

sum_x = 0
sum_y = 0
for i in [30,31,4,25]:
    sum_x += x[i,]
    sum_y += y[i,]
z = np.asarray((sum_x, sum_y))
time = np.linspace(-0.5, 5, x.shape[1])
pl.plot(time, z.T, linewidth=2)
plt.axvline(x=2, linestyle='--')
pl.xlabel('Time (s)', fontsize=14)
pl.ylabel('microvolts', fontsize=14)
pl.legend(['Attend Lead', 'Attend Lag'], loc='best')





