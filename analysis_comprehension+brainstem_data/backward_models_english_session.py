import h5py
import json
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


data_download_dir = "./BrainstemComprehensionData"
preprocessed_data_dir = "./BrainstemComprehensionData"
fname = 'english_session_data_preprocessed.h5'

with open(os.path.join(data_download_dir, 'session_info.json'), 'r') as f:
    session_info = json.load(f)



for condition in ['fM', 'fW']:

    print(condition)


    with h5py.File(os.path.join(preprocessed_data_dir, fname), 'a') as f:
        attended_envelopes = [f[f"{condition}/part{part}/attended"][:] for part in [1,2,3,4]]
        unattended_envelopes = [f[f"{condition}/part{part}/unattended"][:] for part in [1,2,3,4]]
 
    y_train_a = np.hstack(attended_envelopes[:3])[None, :]
    y_test_a = attended_envelopes[3][None, :]

    y_train_u = np.hstack(unattended_envelopes[:3])[None, :]
    y_test_u = unattended_envelopes[3][None, :]

        # now pre-process the eeg
    for participant in session_info['english_session_participants']:

        with h5py.File(os.path.join(preprocessed_data_dir, fname), 'r') as f:
            #import pdb;pdb.set_trace()
            eeg = [f[f'{condition}/part{part}/{participant}'][:, :] for part in [1,2,3,4]]

        X_train = lagged_matrix(np.hstack(eeg[:3]), lags=np.arange(-50,0))
        X_test = lagged_matrix(eeg[3], lags=np.arange(-50,0))

        # fit backward model

        coef_a = np.linalg.inv(X_train @ X_train.T + 1*np.eye(X_train.shape[0])) @ X_train @ y_train_a.T
        y_pred_a = X_test.T @ coef_a

        coef_u = np.linalg.inv(X_train @ X_train.T + 1*np.eye(X_train.shape[0])) @ X_train @ y_train_u.T
        y_pred_u = X_test.T @ coef_u

        print(
            'attended: ',
            np.round(pearsonr(y_pred_a.squeeze(), y_test_a.squeeze())[0], 4),
            'unattended: ',
            np.round(pearsonr(y_pred_u.squeeze(), y_test_u.squeeze())[0], 4)
            )