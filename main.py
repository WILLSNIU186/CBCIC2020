# CBCIC-WCCI-2020
import pandas as pd
import numpy as np
import pdb
import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from data_loader import DataLoader
from preprocessing import Preprocessing

if __name__ == "__main__":

    ########################################################################################################################
    # Import Data
    ########################################################################################################################
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.create_mne_epoch_object()

    ########################################################################################################################
    # Preprocessing
    ########################################################################################################################
    preprocessing = Preprocessing(data_loader)
    sub_id = 2
    filtered_epoch = preprocessing.apply_filter(data=data_loader.train_epochs[sub_id], hi=40, low=None, order=4)
    right_idx = np.where(data_loader.train_label[sub_id, :] == 1)
    left_idx = np.where(data_loader.train_label[sub_id, :] == 2)
    ########################################################################################################################
    # Data Visualization
    ########################################################################################################################
    filtered_epoch.plot_psd()
    filtered_epoch.plot_psd_topomap(normalize=True)

    # define frequencies of interest (log-spaced)
    freqs = np.logspace(*np.log10([1, 40]), num=20)
    n_cycles = freqs / 2.  # different number of cycle per frequency

    power_right, itc_right = tfr_morlet(filtered_epoch[right_idx[0]], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=3, n_jobs=1)
    power_right.plot_topo(baseline=(0, 2), mode='logratio', title='right Average power')
    #
    power_right.plot_topomap(tmin=2.5, tmax=3.5, fmin=13, fmax=30, baseline=(0, 2), mode='logratio', title='right beta', show=False)

    power_left, itc_left = tfr_morlet(filtered_epoch[left_idx[0]], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=3, n_jobs=1)
    power_left.plot_topo(baseline=(0, 2), mode='logratio', title='left Average power', show=False)
    #
    power_left.plot_topomap(tmin=2.5, tmax=3.5, fmin=13, fmax=30, baseline=(0, 2), mode='logratio', title='left beta', show=False)

    power_diff = power_right - power_left

    power_left.plot_topo(baseline=(0, 2), mode='logratio', title='right - left Average power', show=False)

    power_left.plot_topomap(tmin=2.5, tmax=3.5, fmin=13, fmax=30, baseline=(0, 2), mode='logratio',
                            title='Beta right - left', show=True)


    right_evoked = preprocessing.create_evoked_data(filtered_epoch, right_idx)
    right_evoked.plot_topomap(times=[1, 2, 3, 4, 5, 6, 7], time_unit='s', title='right', show=False)
    left_evoked = preprocessing.create_evoked_data(filtered_epoch, left_idx)
    left_evoked.plot_topomap(times=[1, 2, 3, 4, 5, 6, 7], time_unit='s', title='left', show=False)

    mne.combine_evoked([right_evoked, -left_evoked], weights='equal').plot_topomap(times=[1, 2, 3, 4, 5, 6, 7],
                                                                                   time_unit='s', title='right - left',
                                                                                   show=False)
    mne.combine_evoked([-right_evoked, left_evoked], weights='equal').plot_topomap(times=[1, 2, 3, 4, 5, 6, 7],
                                                                                   time_unit='s', title='left - right',
                                                                                   show=True)

    pdb.set_trace()



    ########################################################################################################################
    # Feature extraction
    ########################################################################################################################


    ########################################################################################################################
    # Feature selection
    ########################################################################################################################



    ########################################################################################################################
    # Classification
    ########################################################################################################################
