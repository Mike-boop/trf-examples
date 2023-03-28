import h5py
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

preprocessed_data_dir = "./SurprisalData"

# divide 15 trials into train, val, test parts. Didn't use test parts.
train_parts = range(9)
val_parts = range(9,12)
test_parts = range(12,15)

ridge_parameter=1e3

with h5py.File(os.path.join(preprocessed_data_dir, 'preproc_data.h5'), 'r') as f:

    train_envs = np.hstack([f[f'stim/part{i:02d}'][:] for i in train_parts])[None, :]
    val_envs = np.hstack([f[f'stim/part{i:02d}'][:] for i in val_parts])[None, :]

    ## onset envelopes
    # train_envs = np.diff(train_envs, axis=1, append=0)
    # train_envs[train_envs<0] = 0
    # val_envs = np.diff(val_envs, axis=1, append=0)
    # val_envs[val_envs<0] = 0


    all_coefs = []

    for participant in range(13):

        train_eeg = np.hstack([f[f'eeg/P0{participant:02d}/part{i:02d}'][:] for i in train_parts])
        val_eeg = np.hstack([f[f'eeg/P0{participant:02d}/part{i:02d}'][:] for i in val_parts])


        X_train = lagged_matrix(train_envs, lags=np.arange(-20,50))
        X_val = lagged_matrix(val_envs, lags=np.arange(-20,50))

        coef = np.linalg.inv(X_train@X_train.T + ridge_parameter*np.eye(X_train.shape[0])) @ X_train @ train_eeg.T
        predicted_eeg = X_val.T @ coef

        all_coefs.append(coef)

        t = np.arange(-20, 50)/125
        plt.plot(t, coef, c='grey', lw=1)
        plt.plot(t, np.mean(coef, axis=1), c='red')
        plt.axvline(0, c='black')
        plt.savefig(f'./analysis_surprisal_data/plots/TRF_{participant}.png')
        plt.close()
        print(np.mean([pearsonr(predicted_eeg.T[i], val_eeg[i]) for i in range(63)]))

    mean_coefs = np.mean(all_coefs, axis=0)
    plt.plot(t, mean_coefs, c='grey', lw=1)
    plt.plot(t, np.mean(mean_coefs, axis=1), c='red')
    plt.axvline(0, c='black')
    plt.savefig(f'./analysis_surprisal_data/plots/TRF_mean.png')
    plt.close()