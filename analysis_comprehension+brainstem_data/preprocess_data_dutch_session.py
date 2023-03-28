import os
import json
import h5py
import numpy as np
import mne

from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.stats import zscore
from mne.filter import resample, filter_data

# change these to where you downloaded and uncompressed the data, and where you want to save the preprocessed data.
data_download_dir = "./BrainstemComprehensionData"
preprocessed_data_dir = data_download_dir

mne.set_log_level('CRITICAL')

# find out which sessions the participants took part in
with open(os.path.join(data_download_dir,'session_info.json'), 'r') as f:
    session_info = json.load(f)

output_fname = 'dutch_session_data_preprocessed.h5'


for condition in ['cleanDutch', 'lbDutch', 'mbDutch', 'hbDutch']:

    print(condition)

    with open(os.path.join(data_download_dir, 'audiobooks', condition, 'english_onsets_info.json'), 'r') as f:
        english_onsets_data = json.load(f)

    for part in [1,2,3,4]:

        fs, audio = wavfile.read(os.path.join(data_download_dir, 'audiobooks', condition, f'part_{part}_story.wav'))
        assert fs == 44100
        
        # process the attended speech stream
        attended_envelope = np.abs(hilbert(audio))
        attended_envelope = resample(attended_envelope, 5, 1764) # resample envelope to 125 Hz
        attended_envelope = filter_data(attended_envelope, sfreq=125, l_freq=None, h_freq=50)

        # remove sections where english sentences are playing
        part_onsets_data = english_onsets_data[f'part_{part}']
        dutch_onsets, dutch_offsets = [0], []
            
        for i in range(len(part_onsets_data['onsets'])):

            start_idx = int(part_onsets_data['onsets'][i]*64/44100)
            end_idx = int(part_onsets_data['offsets'][i]*64/44100)

            dutch_offsets.append(start_idx)
            dutch_onsets.append(end_idx)

        dutch_offsets.append(len(attended_envelope))

        # account for 1 second of no babble at start of SiN trials
        if condition in ['lbDutch', 'mbDutch', 'hbDutch']:
            attended_envelope = attended_envelope[125:]

        max_length = len(attended_envelope)

        attended_envelope = np.hstack([attended_envelope[dutch_onsets[i]:dutch_offsets[i]] for i in range(len(dutch_onsets))])

        # save the pre-processed attendeds peech envelope
        with h5py.File(os.path.join(preprocessed_data_dir, output_fname), 'a') as f:
            f.create_dataset(f"{condition}/part{part}/attended", data=zscore(attended_envelope[:max_length]))

        # now pre-process the eeg
        for participant in session_info['dutch_session_participants']:

            with h5py.File(os.path.join(data_download_dir, 'eeg', f'{participant}.h5'), 'r') as f:
                eeg = f[f'{condition}/part_{part}'][:]

            eeg = resample(eeg, 1, 8) # resample 1kHz to 125Hz.
            eeg = filter_data(eeg, sfreq=125, l_freq=0.5, h_freq=8)

            # account for one second of dutch (only) at start of SiN trials
            if condition in ['lbDutch', 'mbDutch', 'hbDutch']:
                eeg = zscore(eeg[:, 125:125+max_length], axis=1)
            else:
                eeg = zscore(eeg[:, :max_length], axis=1)

            eeg = np.hstack([eeg[:, dutch_onsets[i]:dutch_offsets[i]] for i in range(len(dutch_onsets))])
            print(attended_envelope.shape, eeg.shape)

            with h5py.File(os.path.join(preprocessed_data_dir, output_fname), 'a') as f:
                f.create_dataset(f"{condition}/part{part}/{participant}", data=eeg)