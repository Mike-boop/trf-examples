import h5py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def lagged_matrix(X, lags=[0]):
    '''
    X: shape (n_features, n_samples)
    lags: the number of lags to apply to the samples axis
    returns: the toeplitz matrix required for OLS system identification
    '''
    rows = []

    for lag in lags:
        rows.append(np.roll(X, lag, axis=1))

    return np.vstack(rows)


data_download_dir = "./BrainstemComprehensionData"
preprocessed_data_dir = "./BrainstemComprehensionData"
fname = 'dutch_session_data_preprocessed.h5'

with open(os.path.join(data_download_dir, 'session_info.json'), 'r') as f:
    session_info = json.load(f)


for condition in ['cleanDutch', 'lbDutch', 'mbDutch', 'hbDutch']:

    print(condition)
    all_coefs = []


    with h5py.File(os.path.join(preprocessed_data_dir, fname), 'a') as f:
        attended_envelopes = [f[f"{condition}/part{part}/attended"][:] for part in [1,2,3,4]]

    X_train = np.hstack(attended_envelopes[:3])[None, :]
    X_test = attended_envelopes[3][None, :]

    # onset envelopes
    X_train = np.diff(X_train, axis=1, append=0)
    X_train[X_train<0] = 0
    X_test = np.diff(X_test, axis=1, append=0)
    X_test[X_test<0] = 0

    X_train = lagged_matrix(X_train, lags=np.arange(-20,50))
    X_test = lagged_matrix(X_test, lags=np.arange(-20,50))

    # now pre-process the eeg
    for participant in session_info['dutch_session_participants']:

        with h5py.File(os.path.join(preprocessed_data_dir, fname), 'r') as f:
            #import pdb;pdb.set_trace()
            eeg = [f[f'{condition}/part{part}/{participant}'][:, :] for part in [1,2,3,4]]

        y_train = np.hstack(eeg[:3])
        y_test = eeg[3]

        # fit backward model

        coef = np.linalg.inv(X_train @ X_train.T + 1e3*np.eye(X_train.shape[0])) @ X_train @ y_train.T
        y_pred = X_test.T @ coef
        all_coefs.append(coef)

        print(np.mean([pearsonr(y_pred.T[i], y_test[i]) for i in range(63)]))
    

    t = np.arange(-20, 50)/125
    mean_coefs = np.mean(all_coefs, axis=0)
    plt.plot(t, mean_coefs, c='grey', lw=1)
    plt.plot(t, np.mean(mean_coefs, axis=1), c='red')
    plt.axvline(0, c='black')
    plt.savefig(f'./analysis_comprehension+brainstem_data/plots/TRF_mean_{condition}.png')
    plt.close()