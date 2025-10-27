import os
import sys
from sklearn.model_selection import train_test_split

import h5py as h5
import numpy as np

def load_data(path: str, matgendata=False):
    try:
        file_handle = h5.File(path,'r+')
    except ValueError:
        print(f"Error: can't open HDF5 file '{path}' for reading (it might be malformed) ...")
        sys.exit(-1)
    
    if matgendata:
        x = file_handle['X'][:]
        x = x.reshape(x.shape[0], 1024, 1, 2) 
        y = file_handle['Y'][:]
        z = file_handle['Z'][:]
    
    else:
        x = file_handle['all_IQ_int8'][:]
        x = x.reshape(x.shape[0], 1024, 1, 2) 
        y = file_handle['all_labels'][:]
        z = file_handle['all_SNRs'][:]
        
    x, y, z = usable_snr_data(x, y, z)
    return train_test_split(x, y, z, test_size=0.2, random_state=0)

def val_data_split(X, Y, Z):
    return train_test_split(X, Y, Z, test_size = 0.5, random_state=0)

def best_snr_data(X, Y, Z):
    best_snr_indices = np.where(np.any(Z == 30, axis=1))
    return X[best_snr_indices], Y[best_snr_indices], Z[best_snr_indices]

def usable_snr_data(X, Y, Z):
    usable_snr_indices = np.where(np.any(Z >= 6, axis = 1))
    return X[usable_snr_indices], Y[usable_snr_indices], Z[usable_snr_indices]