import h5py
import os
import numpy as np
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

ridge_parameter=1

with h5py.File(os.path.join(preprocessed_data_dir, 'preproc_data.h5'), 'r') as f:

    train_envs = np.hstack([f[f'stim/part{i:02d}'][:] for i in train_parts])[None, :]
    val_envs = np.hstack([f[f'stim/part{i:02d}'][:] for i in val_parts])[None, :]


    for participant in range(13):

        train_eeg = np.hstack([f[f'eeg/P0{participant:02d}/part{i:02d}'][:] for i in train_parts])
        val_eeg = np.hstack([f[f'eeg/P0{participant:02d}/part{i:02d}'][:] for i in val_parts])


        X_train = lagged_matrix(train_eeg, lags=np.arange(-50,0))
        X_val = lagged_matrix(val_eeg, lags=np.arange(-50,0))

        coef = np.linalg.inv(X_train@X_train.T + ridge_parameter*np.eye(X_train.shape[0])) @ X_train @ train_envs.T

        predicted_envs = X_val.T @ coef
        print(pearsonr(predicted_envs.squeeze(), val_envs.squeeze()))