import os
import numpy as np
from HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf

bp_low = 0.5
bp_upp = 6
E_val = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

parent_dir = os.getcwd()
path = os.path.join(parent_dir, 'subject_name')
subject_name = 'K114'
session_name = '001_BCI_TRN'

time_index = np.linspace(0, 800, 25)

eeg_trunc_obj = sio.loadmat('/Users/Tej/Documents/GitHub/BIOS-584/data/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat')
eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = eeg_trunc_obj['Type']

K114_output = produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, E_val)

plot_trunc_mean(
    eeg_tar_mean=K114_output[0],
    eeg_ntar_mean=K114_output[1],
    subject_name=subject_name,
    time_index=time_index,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls,
    y_limit=[-5, 8],
    fig_size=(12, 12)
)

plot_trunc_cov(
    eeg_cov=K114_output[2],        # Target covariance location
    cov_type="Target",
    time_index=time_index,
    subject_name=subject_name,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls
)

plot_trunc_cov(
    eeg_cov=K114_output[3],        # non-target covariance location
    cov_type="Non-Target",
    time_index=time_index,
    subject_name=subject_name,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls
)

plot_trunc_cov(
    eeg_cov=K114_output[4],        # all covariance location
    cov_type="All",
    time_index=time_index,
    subject_name=subject_name,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls
)