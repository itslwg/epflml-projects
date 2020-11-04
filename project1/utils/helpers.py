# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv

def import_data(path):
    """
    Import csv files and return array of X,y and vector of the column names.
    """
    train = np.loadtxt(
        f"{path}train.csv",
        delimiter = ",",
        skiprows=0,
        dtype=str
    )

    test = np.loadtxt(
        f"{path}test.csv",
        delimiter = ",",
        skiprows=0,
        dtype=str
    )

    col_names = train[0,:]

    # Remove column names
    train = np.delete(train, obj=0, axis=0)
    test = np.delete(test, obj=0, axis=0)

    # Map 0 & 1 to label
    label_idx = np.where(col_names == "Prediction")[0][0]
    train[:,label_idx] = np.where(train[:,label_idx]=="s", 1, 0)

    test[:,label_idx] = 0

    # Replace -999 with nan
    train = train.astype(np.float32)
    train[train == -999] = np.nan

    test = test.astype(np.float32)
    test[test == -999] = np.nan
    return train, test, col_names

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def build_poly(x, degree):
    """polynomial basis functions for each column of x, for j=1 up to j=degree, and single constant term."""
    if (degree < 0): raise ValueError("degree must be positive")
    
    phi = np.empty((x.shape[0], x.shape[1]*degree+1))
    
    # Constant term
    phi[:,-1] = 1
    
    # Higher order terms
    for j in range(x.shape[1]):
        phi[:,j*degree] = x[:,j]
        for d in range(1,degree):
            col = j*degree+d
            phi[:,col] = phi[:,col-1] * x[:,j]
    
    return phi

def prepare_features(tx_nan, degree, mean=None): 
    # Get column means, if necessary
    if mean is None:
        mean = np.nanmean(tx_nan,axis=0)
    
    # Replace NaNs
    tx_val = np.where(np.isnan(tx_nan), mean, tx_nan)
    
    # Polynomial features
    tx = build_poly(tx_val, degree)
    const_col = tx.shape[1]-1
    
    # Add NaN indicator columns
    nan_cols = np.flatnonzero(np.any(np.isnan(tx_nan), axis=0))

    ind_cols = np.empty((tx_nan.shape[0], nan_cols.shape[0]))
    ind_cols = np.where(np.isnan(tx_nan[:,nan_cols]), 1, 0)

    tx = np.c_[tx, ind_cols]
    
    # Standardize
    tx, _, _ = standardize_numpy(tx)
    tx[:,const_col] = 1.0
    
    return tx, mean, nan_cols

def sigmoid(x):
    """


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        apply sigmoid.

    """
    epsilon = 1E-12
    a = 1 / (1 + np.exp(-x))
    a = np.where(np.isclose(a, 0.0), epsilon, a)
    a = np.where(np.isclose(a, 1.0), (1-epsilon), a)
    return a

def standardize_numpy(x, mean=None, std=None):
    """Standardize the original data set. Works on numpy arrays."""
    if mean is None: mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    if std is None: std = x.std(axis=0, keepdims=True)
    x = x / std
    return x, mean, std

def predict_reg_log_regression(tx, w):
    return np.rint(sigmoid(tx @ w))
