'''
Use this script to preprocess the EEG and audiobook files
Will resample EEG and envelopes to 125 Hz. 
EEG will be filtered between 0.5 - 8 Hz; envelopes will be filtered below 50 Hz.
'''

import h5py
import glob
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.stats import zscore
from mne.filter import filter_data, resample

# change these to where you downloaded and uncompressed the data, and where you want to save the preprocessed data.
data_download_dir = "./SurprisalData"
preprocessed_data_dir = data_download_dir

for i, stimulus_file in enumerate(sorted(glob.glob(os.path.join(data_download_dir, 'audiobooks/', '*.wav')))):

    chapter_name = os.path.basename(stimulus_file).replace('.wav', '')
    fs, audio = wavfile.read(stimulus_file)

    assert fs == 16000

    envelope = np.abs(hilbert(audio))
    envelope = resample(envelope, 1, 128) # resample to 125
    envelope = filter_data(envelope, sfreq=125, l_freq=None, h_freq=50)

    with h5py.File(os.path.join(preprocessed_data_dir, 'preproc_data.h5'), 'a') as f:

        f.create_dataset(f'stim/part{i:02d}', data=zscore(envelope))

    for participant in range(13):

        with h5py.File(os.path.join(data_download_dir, f'P{participant:02d}.h5'), 'r') as f:

            eeg = f[f'data/{chapter_name}'][:]

        eeg = resample(eeg, 1, 8) # resample 1kHz to 125Hz.
        eeg = filter_data(eeg, sfreq=125, l_freq=0.5, h_freq=8)
        eeg = zscore(eeg[:, :len(envelope)], axis=1)

        with h5py.File(os.path.join(preprocessed_data_dir, 'preproc_data.h5'), 'a') as f:

            f.create_dataset(f'eeg/P0{participant:02d}/part{i:02d}', data=eeg)