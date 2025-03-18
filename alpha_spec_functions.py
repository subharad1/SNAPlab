import numpy 
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

def calculate_alpha_power(tfr):
    alpha_power = numpy.zeros((len(tfr), 32))
    for epoch in range(len(tfr)):
        alpha_power_epoch = tfr.data[epoch, :, :, :]
        alpha_mean_epoch = alpha_power_epoch.mean(axis=1).mean(axis=1)
        alpha_power[epoch, :] = alpha_mean_epoch
    return alpha_power.mean(axis=0)

def calculate_normalized_difference(left_power, right_power):
    return ((left_power - right_power) / (left_power + right_power)) * 0.5 * 100