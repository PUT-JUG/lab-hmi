import pandas as pd
import scipy.signal as signal
import numpy as np

def  filter_emg(data:np.array, fs:int, Rs:int, notch:bool):
    
    N, fn = signal.ellipord([46.0,54.0], [47.0,53], .01, Rs, fs=fs)
    be, ae = signal.ellip(N, .01, Rs,fn, fs=fs,btype='bandstop')
    N, fn = signal.ellipord([96,104.0], [97,103], .01, Rs, fs=fs)
    be_100, ae_100 = signal.ellip(N, .01, Rs,fn, fs=fs,btype='bandstop')
    N, fn = signal.cheb2ord(15, 10, .0086, 55, fs=500)
    bb, ab = signal.cheby2(N, 50,fn, 'high', fs=fs)
    signal_filtered = signal.lfilter(be_100,ae_100,signal.lfilter(be,ae,signal.lfilter(bb,ab,data)))
#     signal_filtered = data
    signal_filtered_zero_ph = signal.filtfilt(be_100,ae_100,signal.filtfilt(be,ae,signal.filtfilt(bb,ab,data)))
    return signal_filtered, signal_filtered_zero_ph 
    
def subsample_emg(data:np.array, fs:int, r:int, Rs:int):
    return data[::r]
    
def filter_force(data:np.array, fs:int):
    signal_filtered = data
    signal_filtered_zero_ph = data
    return signal_filtered, signal_filtered_zero_ph 
