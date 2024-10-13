#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:39:43 2024

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

# %% 
fpath = '/Users/sharadhib/Library/CloudStorage/Box-Box/SNAPlab/results/'
subjects = ['SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP008' , 'HR']

# %% Evoked
evoked_lefts = [f'{subject_id}_evoked_left_ave.fif' for subject_id in subjects]
evoked_rights = [f'{subject_id}_evoked_right_ave.fif' for subject_id in subjects]

evoked_left = [mne.read_evokeds(lefts)[0] for lefts in evoked_lefts]
evoked_right = [mne.read_evokeds(rights)[0] for rights in evoked_rights]

average_evoked_left = mne.grand_average(evoked_left)
average_evoked_right = mne.grand_average(evoked_right)

x = average_evoked_left.data*1e6
y = average_evoked_right.data*1e6

# %%
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
pl.legend(['Attend Left', 'Attend Right'], loc='best')


# %% Evoked During
evoked_lefts_dur = [f'{subject_id}_evoked_left_dur_ave.fif' for subject_id in subjects]
evoked_rights_dur = [f'{subject_id}_evoked_right_dur_ave.fif' for subject_id in subjects]

evoked_left_dur = [mne.read_evokeds(lefts)[0] for lefts in evoked_lefts_dur]
evoked_right_dur = [mne.read_evokeds(rights)[0] for rights in evoked_rights_dur]

average_evoked_left_dur = mne.grand_average(evoked_left_dur)
average_evoked_right_dur = mne.grand_average(evoked_right_dur)

x = average_evoked_left_dur.data*1e6
y = average_evoked_right_dur.data*1e6

# %%
sum_x = 0
sum_y = 0
for i in [30,31,4,25]:
    sum_x += x[i,]
    sum_y += y[i,]
z = np.asarray((sum_x, sum_y))
time = np.linspace(2, 5, x.shape[1])
pl.plot(time, z.T, linewidth=2)
pl.xlabel('Time (s)', fontsize=14)
pl.ylabel('microvolts', fontsize=14)
pl.legend(['Attend Left', 'Attend Right'], loc='best')

# %% Evoked Left Lead vs Right Lag 
evoked_lefts_lead = [f'{subject_id}_evoked_left_lead_ave.fif' for subject_id in subjects]
evoked_rights_lag = [f'{subject_id}_evoked_right_lag_ave.fif' for subject_id in subjects]

evoked_left_lead = [mne.read_evokeds(lefts)[0] for lefts in evoked_lefts_lead]
evoked_right_lag = [mne.read_evokeds(rights)[0] for rights in evoked_rights_lag]

average_evoked_left_lead = mne.grand_average(evoked_left_lead)
average_evoked_right_lag = mne.grand_average(evoked_right_lag)

x = average_evoked_left_lead.data*1e6
y = average_evoked_right_lag.data*1e6

sum_x = 0
sum_y = 0
for i in [30,31,4,25]:
    sum_x += x[i,]
    sum_y += y[i,]
z = np.asarray((sum_x, sum_y))
time = np.linspace(-0.5, 5, x.shape[1])
pl.plot(time, z.T)
#plt.axvline(x=2, linestyle='--', label='Left onset')
#plt.axvline(x=2.25, linestyle='--', color='orange', label='right inset')
pl.xlabel('Time (s)', fontsize=12)
pl.ylabel('Evoked Potential (µV)', fontsize=12)
pl.legend(['Attend Left', 'Attend Right'], loc='best')

# %% Load spectrogram data
file = '_left_right_pre.npy'
file_dur = '_left_right_dur.npy'

sum_left_right_pre = np.zeros(32)
sum_left_right_dur = np.zeros(32)
for subj in subjects:
    print(fpath+subj+file)
    left_right_pre = np.load(fpath+subj+file)
    left_right_dur = np.load(fpath+subj+file_dur)
    sum_left_right_pre += left_right_pre
    sum_left_right_dur += left_right_dur
    
avg_left_right_pre = sum_left_right_pre/len(subjects)
avg_left_right_dur = sum_left_right_dur/len(subjects)

# %% Load info
raw, eves = bs.importbdf('/Users/sharadhib/Library/CloudStorage/Box-Box/SNAPlab/SP003_Att+001.bdf', hptsname='/Users/sharadhib/Library/CloudStorage/Box-Box/SNAPlab/biosemi10_20.hpts')
raw.notch_filter(np.arange(60, 241, 60), filter_length='auto', fir_design='firwin')
raw.filter(1, 20)

blinks = find_blinks(raw, ch_name=['Fp1'])
epochs_blinks = mne.Epochs(raw, blinks, event_id=998, baseline=(-0.25, 0.25),
                             reject=dict(eeg=500e-6), tmin=-0.25, tmax=0.25)
blink_proj = compute_proj_epochs(epochs_blinks, n_eeg=3)
raw.add_proj(blink_proj)
raw.del_proj(1)
epochs_left = mne.Epochs(raw, eves, event_id=[7,8], baseline=(-0.5, 0), proj=True,
                        tmin=-0.5, tmax=5, reject=dict(eeg=200e-6))
# %%

channels = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3',
 'O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']

epoch_pre_left = epochs_left.copy().load_data().crop(tmin=0.0, tmax=2.0)
epoch_dur_left = epochs_left.copy().load_data().crop(tmin=2.0, tmax=5.0)

info_pre_picked = mne.pick_info(epochs_left.info, mne.pick_channels(epochs_left.info['ch_names'], include =channels))
info_dur_picked = mne.pick_info(epochs_left.info, mne.pick_channels(epochs_left.info['ch_names'], include =channels))

# %%
fig, ax = plt.subplots(figsize=(8, 6))
mne.viz.plot_topomap(avg_left_right_pre, info_pre_picked, names=channels,  axes=ax, ch_type='eeg',show=True)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
mne.viz.plot_topomap(avg_left_right_dur, info_pre_picked, names=channels,  axes=ax, ch_type='eeg',show=True)