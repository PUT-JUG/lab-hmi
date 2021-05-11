import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix


def  rms(signal, window=500, stride=100, fs=5120, columns_emg=['EMG_8', 'EMG_9']):# wartści długości okna i przesunięcia w [ms]
    
    return signal[columns_emg].iloc[int(window/1000*fs)::int(stride/1000*fs)]
    
def zc(signal, threshold:float=0.1, window:float=500, stride:float=100, fs=5120, columns_emg=['EMG_8', 'EMG_9']):# wartści długości okna i przesunięci[column columns_emg].iloc[]
    return signal[columns_emg].iloc[int(window/1000*fs)::int(stride/1000*fs)]
    

def find_threshold(signal, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id = 0):
    return  signal.loc[signal[column_gesture]==idle_gesture_id, columns_emg].mean()

def norm_emg(signal, norm_coeffs, columns_emg=['EMG_8', 'EMG_9']):
    coeffs = norm_coeffs[columns_emg]
    return signal[columns_emg]
    
def k_mean(features, y_true):
    y_pred=y_true
    return  confusion_matrix(y_true, y_pred)
