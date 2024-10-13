#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:15:43 2024

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
subj = 'SP009'
resultsdir = '/Users/sharadhib/Library/CloudStorage/Box-Box/SNAPlab/results/'

lefts = []
rights = []

channels_to_drop = ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

bdfs = fnmatch.filter(os.listdir(fpath), subj + '_Att*.bdf')

if subj == 'SP009':
    bdfs = ['SP009_AttentionTask.bdf',
     'SP009_AttentionTask+001.bdf']

for k, bdf in enumerate(bdfs):
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

    events_left = [7,8]
    events_left_use = [event_left for event_left in events_left if event_left in eves[:,2]]
    
    events_right = [9,10]
    events_right_use = [event_right for event_right in events_right if event_right in eves[:,2]]

    if events_left_use:
        epochs_left = mne.Epochs(raw, eves, event_id=events_left_use, baseline=(-0.5, 0.0), proj=True,
                        tmin=-0.5, tmax=5, reject=dict(eeg=200e-6))
    if events_right_use:
        epochs_right = mne.Epochs(raw, eves, event_id=events_right_use, baseline=(-0.5, 0.0), proj=True,
                        tmin=-0.5, tmax=5, reject=dict(eeg=200e-6))
    lefts += [epochs_left, ]
    rights += [epochs_right, ]

left = concatenate_epochs(lefts)
right = concatenate_epochs(rights)    

# %%
raw.plot_projs_topomap()

# %%
epoch_name_l = resultsdir+subj + '_epochs_left-epo.fif'
epoch_name_r = resultsdir+subj + '_epochs_right-epo.fif'

left.save(epoch_name_l,overwrite=True)
right.save(epoch_name_r, overwrite=True)
# %%
evoked_left = left.average()
evoked_right = right.average()

fname_left = resultsdir+subj + '_evoked_left_ave.fif'
fname_right = resultsdir+subj + '_evoked_right_ave.fif'

evoked_left.save(fname_left,overwrite=True)
evoked_right.save(fname_right, overwrite=True)

x = evoked_left.data * 1e6
t = evoked_right.times
y = evoked_right.data * 1e6
z = np.asarray((x[30,], y[30,]))
pl.plot(t, z.T, linewidth=2)
pl.xlabel('Time (s)', fontsize=14)
pl.ylabel('microvolts', fontsize=14)
pl.legend(['Attend Left', 'Attend Right'], loc='best')
pl.show()

# %%
epoch_left_dur = left.copy().crop(tmin=2.0, tmax=5.0)
epoch_right_dur = right.copy().crop(tmin=2.0, tmax=5.0)

evoked_left_dur = epoch_left_dur.average()
evoked_right_dur = epoch_right_dur.average()

fname_left = resultsdir+subj + '_evoked_left_dur_ave.fif'
fname_right = resultsdir+subj + '_evoked_right_dur_ave.fif'

evoked_left_dur.save(fname_left,overwrite=True)
evoked_right_dur.save(fname_right, overwrite=True)

x = evoked_left_dur.data * 1e6
t = evoked_left_dur.times
y = evoked_right_dur.data * 1e6
z = np.asarray((x[30,], y[30,]))
pl.plot(t, z.T, linewidth=2)
pl.xlabel('Time (s)', fontsize=14)
pl.ylabel('microvolts', fontsize=14)
pl.legend(['Attend Left', 'Attend Right'], loc='best')
pl.show()

# %%
epoch_left_pre = left.copy().crop(tmin=0.0, tmax=2.0)
epoch_right_pre = right.copy().crop(tmin=0.0, tmax=2.0)

# %%
freqs = np.arange(7., 16., 1.)
n_cycles = freqs * 0.4
# %%
power_alpha_left_pre = tfr_multitaper(epoch_left_pre, freqs, n_cycles, return_itc=False, picks=['eeg'])
power_alpha_right_pre = tfr_multitaper(epoch_right_pre, freqs, n_cycles, return_itc=False, picks=['eeg'])

fname_pleft = resultsdir+subj+ '_left_alpha_pre_new_tfr.h5'
fname_pright = resultsdir+subj+ '_right_alpha_pre_new_tfr.h5'

power_alpha_left_pre.save(fname_pleft, overwrite=True)
power_alpha_right_pre.save(fname_pright, overwrite=True)

# %%
avg_power_alpha_left_pre = power_alpha_left_pre.data.mean(axis=1).mean(axis=1)
avg_power_alpha_right_pre = power_alpha_right_pre.data.mean(axis=1).mean(axis=1)
# %%
norm_avg_power_alpha_left_right_pre = (avg_power_alpha_left_pre - avg_power_alpha_right_pre)/(avg_power_alpha_left_pre + avg_power_alpha_right_pre)*0.5*100 

fname_left_right_pre = resultsdir+subj+ '_norm_left_right_pre'
np.save(fname_left_right_pre, norm_avg_power_alpha_left_right_pre)
# %%
channels = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3',
 'O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']
info_picked = mne.pick_info(epoch_left_pre.info, mne.pick_channels(epoch_left_pre.info['ch_names'], include =channels))

fig, ax = plt.subplots(figsize=(8, 6))
mne.viz.plot_topomap(norm_avg_power_alpha_left_right_pre, info_picked, names=channels,  axes=ax, ch_type='eeg',show=True)
# %%
power_alpha_left_dur = tfr_multitaper(epoch_left_dur, freqs, n_cycles, return_itc=False, picks=['eeg'])
power_alpha_right_dur = tfr_multitaper(epoch_right_dur, freqs, n_cycles, return_itc=False, picks=['eeg'])

fname_pleft = resultsdir+subj+ '_left_alpha_dur_new_tfr.h5'
fname_pright = resultsdir+subj+ '_right_alpha_dur_new_tfr.h5'

power_alpha_left_dur.save(fname_pleft, overwrite=True)
power_alpha_right_dur.save(fname_pright, overwrite=True)
# %%
avg_power_alpha_left_dur = power_alpha_left_dur.data.mean(axis=1).mean(axis=1)
avg_power_alpha_right_dur = power_alpha_right_dur.data.mean(axis=1).mean(axis=1)
# %%
norm_avg_power_alpha_left_right_dur = (avg_power_alpha_left_dur - avg_power_alpha_right_dur)/(avg_power_alpha_left_dur + avg_power_alpha_right_dur)*0.5*100

fname_left_right_dur = resultsdir+subj+ '_norm_left_right_dur'
np.save(fname_left_right_dur, norm_avg_power_alpha_left_right_dur)
# %%
channels = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3',
 'O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']
info_picked = mne.pick_info(epoch_left_dur.info, mne.pick_channels(epoch_left_dur.info['ch_names'], include =channels))

fig, ax = plt.subplots(figsize=(8, 6))
mne.viz.plot_topomap(norm_avg_power_alpha_left_right_dur, info_picked, names=channels,  axes=ax, ch_type='eeg',show=True)


