import numpy as np
from scipy import io

import mne

class DataLoader():

    def __init__(self):
        self.base_folder = "D:\OneDrive - University of Waterloo\Jiansheng\CBCIC2020\Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow-master\\"
        self.sub_ids = list(range(0, 8))
        self.evaluate_sub_ids = [9, 10]
        self.fs = 512
        self.num_ch = 12
        self.trial_len = 8
        self.num_trial_train = 80
        self.num_trial_test = 40
        self.num_subject = len(self.sub_ids)
        self.num_evaluate_subject = len(self.evaluate_sub_ids)

        self.channel_dict = {'F3': 0, 'FC3': 1, 'C3': 2, 'CP3': 3, 'P3': 4, 'FCz': 5, 'CPz': 6, 'F4': 7, 'FC4': 8,
                             'C4': 9, 'CP4': 10, 'P4': 11}
        self.channel_names = list(self.channel_dict.keys())
        self.channel_position_dict = {'F3': 1, 'FC3': 4, 'C3': 7, 'CP3': 10, 'P3': 13, 'FCz': 5, 'CPz': 11, 'F4': 3,
                                 'FC4': 6, 'C4': 9, 'CP4': 12, 'P4': 15}
        self.channel_type = 'eeg'

        self.train_data = np.zeros((self.num_subject, self.num_trial_train, self.num_ch, self.trial_len * self.fs),
                                   dtype=float)
        self.train_label = np.zeros((self.num_subject, self.num_trial_train), dtype=float)
        self.test_data = np.zeros((self.num_subject, self.num_trial_test, self.num_ch, self.trial_len * self.fs),
                                  dtype=float)
        self.evaluate_data = np.zeros((self.num_evaluate_subject, self.num_trial_test, self.num_ch,
                                       self.trial_len * self.fs), dtype=float)


    def load_data(self):
        for sub_id in self.sub_ids:
            self.raw_eeg_train = io.loadmat(self.base_folder + "parsed_P0{}T.mat".format(sub_id + 1))
            self.train_label[sub_id, :] = np.squeeze(self.raw_eeg_train["Labels"])
            self.train_data[sub_id, :, :, :] = self.raw_eeg_train["RawEEGData"]/10**6

            self.raw_eeg_test = io.loadmat(self.base_folder + "parsed_P0{}E.mat".format(sub_id + 1))
            self.test_data[sub_id, :, :, :] = self.raw_eeg_test["RawEEGData"]/10**6

        for evaluate_sub_id in self.evaluate_sub_ids:
            self.raw_eeg_evaluate = io.loadmat(self.base_folder + "parsed_P0{}E.mat".format(sub_id))
            self.evaluate_data[evaluate_sub_id-9, :, :, :] = self.raw_eeg_evaluate["RawEEGData"]/10**6

    def create_mne_epoch_object(self):
        info = mne.create_info(sfreq = self.fs, ch_names = self.channel_names, ch_types=self.channel_type)
        info.set_montage('standard_1020')
        self.train_epochs = dict([(key, []) for key in self.sub_ids])
        for sub_id in self.sub_ids:
            self.train_epochs[sub_id] = mne.EpochsArray(data=self.train_data[sub_id, :, :, :], info=info)

        self.test_epochs = dict([(key, []) for key in self.sub_ids])
        for sub_id in self.sub_ids:
            self.test_epochs[sub_id] = mne.EpochsArray(data=self.test_data[sub_id, :, :, :], info=info)

        self.test_epochs = dict([(key, []) for key in [0, 1]])
        for sub_id in [0, 1]:
            self.test_epochs[sub_id] = mne.EpochsArray(data=self.evaluate_data[sub_id, :, :, :], info=info)


