import numpy as np
import csv

# =============================================================================
# Models
# =============================================================================

def least_squares_GD(y, tx, initial_w,
                     max_iters, gamma, verbose=False):
    """Least squares with MSE loss and Gradient Descent."""
    ws = [initial_w]
    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        # Compute gradient and loss
        g = mse_grad(y, tx, w)
        loss = mse(y, tx, w)
        # Update the weights
        w = w - gamma * g
        w = w.ravel()
        # Store the weights and loss
        ws.append(w)
        losses.append(loss)

        if verbose:
            print("Gradient descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w,
                      max_iters, gamma, verbose=False):
    """Least squares with MSE loss and Stochasitc Gradient Descent."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            # Compute gradient and loss
            g = mse_grad(minibatch_y, minibatch_tx, w)
            loss = mse(minibatch_y, minibatch_tx, w)
            # Update the weights
            w = w - gamma * g
            w = w.ravel()
            # store w and loss
            ws.append(w)
            losses.append(loss)
            if verbose:
                print("Stochastic gradient descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], mse(y, tx, ws[-1])


def least_squares(y, tx):
    """Linear regression fit using normal equations."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Ridge regression fit using normal equations """
    a = (tx.T @ tx) + lambda_*2*tx.shape[0] * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters,
                        gamma, verbose=False):
    """Logistic regression with log loss and Gradient Descent."""
    ws = [initial_w]
    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        # Compute gradient and loss
        g = logistic_grad(y, tx, w)
        loss = logistic_error(y, tx, w)
        # Update the weights
        w = w - gamma * g
        w = w.ravel()
        # Store the weights and loss
        ws.append(w)
        losses.append(loss)

        if verbose:
            print("Gradient descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, reg, initial_w,
                            max_iters, gamma, verbose=False,
                            early_stopping=True, tol = 0.0001,
                            patience = 5):
    """Regularized logistic regression with log loss and Gradient Descent with early stopping"""
    ws = [initial_w]
    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        # Compute gradient and loss
        g = reg_logistic_grad(y, tx, w, lambda_, reg)
        loss = reg_logistic_error(y, tx, w, lambda_, reg)

        # Update the weights
        w = w - gamma * g
        w = w.ravel()
        # Store the weights and loss
        ws.append(w)
        losses.append(loss)

        if verbose:
            print("Gradient descent({bi}/{ti}): loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss))

        # Early stopping
        if (early_stopping) and (n_iter > patience):
            # Check if loss has improved by tol in last patience iters
            l_pat = reg_logistic_error(y, tx, ws[-patience], lambda_, reg)
            l_1 = reg_logistic_error(y, tx, ws[-1], lambda_, reg)
            if ((l_pat - l_1) < tol):
                print(f"Stopped after {n_iter} it.")
                break

    return ws[-1], losses[-1]

# =============================================================================
# Cost functions
# =============================================================================

def mse(y, tx, w):
    """Mean squared error loss function."""
    e = y - tx @ w
    return (1/(2*tx.shape[0])) * np.sum(e**2)

def logistic_error(y, tx, w):
    """Log loss function."""
    a = sigmoid(tx @ w)
    loss = - (1 / tx.shape[0]) * np.sum((y * np.log(a)) + ((1 - y) * np.log(1 - a)))
    return loss

def reg_logistic_error(y, tx, w, lambda_, reg):
    """Log loss function with regularization term."""
    assert (reg==1 or reg==2), "reg needs to be 1 or 2"
    loss = logistic_error(y, tx, w) + lambda_ * (np.linalg.norm(w, reg) ** reg)
    return loss

# =============================================================================
# Gradients
# =============================================================================

def mse_grad(y, tx, w):
    """Compute gradient for MSE loss."""
    e = y - tx @ w
    return (-1/tx.shape[0]) * tx.T @ e

def logistic_grad(y, tx, w):
    """Compute gradient for log loss."""
    e = sigmoid(tx @ w) - y
    return (1/tx.shape[0]) * tx.T @ e

def reg_logistic_grad(y, tx, w, lambda_, reg):
    """Compute gradient for log loss with regularization."""
    assert (reg==1 or reg==2), "reg needs to be 1 or 2"
    if (reg==1):
        # L1 regularization
        return logistic_grad(y, tx, w) + lambda_ * np.sign(w)
    else:
        # L2 regularization
        return logistic_grad(y, tx, w) + 2 * lambda_ * w

# =============================================================================
# Activation functions
# =============================================================================

def sigmoid(x):
    """Compute sigmoid function."""
    epsilon = 1E-12
    a = 1 / (1 + np.exp(-x))
    a = np.where(np.isclose(a, 0.0), epsilon, a)
    a = np.where(np.isclose(a, 1.0), (1-epsilon), a)
    return a

# =============================================================================
# Helpers
# =============================================================================

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    # Please note this code was provided to us during the lab sessions.
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

def import_data(path="data/"):
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

def create_csv_submission(ids, y_pred, name):
    # Please note this code was provided to us during the lab sessions.
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

def standardize_numpy(x, mean=None, std=None):
    """Standardize the original data set. Works on numpy arrays."""
    if mean is None: mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    if std is None: std = x.std(axis=0, keepdims=True)
    x = x / std
    return x, mean, std

# =============================================================================
# Prepare features
# =============================================================================

def split_X_y(train, test, cols):
    """Create tx matrix for train & test + y vector for train."""
    idx_id = np.where(cols=="Id")[0][0]
    idx_pred = np.where(cols=="Prediction")[0][0]

    tx_train = np.delete(train, [idx_id, idx_pred], axis=1)
    y_train = train[:,idx_pred].copy()
    tx_test = np.delete(test, [idx_id, idx_pred], axis=1)

    return tx_train, y_train, tx_test

def build_poly(x, degree):
    """Polynomial basis functions for each column of x, for j=1 up to j=degree, and single constant term."""
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

def prepare_features(tx_nan, degree, mean_nan=None, mean=None, std=None):
    """Clean and prepare for learning.  Mean imputing, missing value indicator, standardize."""
    # Get column means, if necessary
    if mean_nan is None: mean_nan = np.nanmean(tx_nan,axis=0)

    # Replace NaNs
    tx_val = np.where(np.isnan(tx_nan), mean_nan, tx_nan)

    # Polynomial features
    tx = build_poly(tx_val, degree)
    const_col = tx.shape[1]-1

    # Add NaN indicator columns
    nan_cols = np.flatnonzero(np.any(np.isnan(tx_nan), axis=0))

    ind_cols = np.empty((tx_nan.shape[0], nan_cols.shape[0]))
    ind_cols = np.where(np.isnan(tx_nan[:,nan_cols]), 1, 0)

    tx = np.c_[tx, ind_cols]

    # Standardize
    tx, mean, std = standardize_numpy(tx, mean, std)
    tx[:,const_col] = 1.0

    return tx, mean, std, mean_nan, nan_cols

# =============================================================================
# Performance metrics
# =============================================================================

def logistic_prediction(tx, w):
    """Make a prediction with logistic regression model."""
    return np.rint(sigmoid(tx @ w))

def regression_prediction(tx, w):
    """Make a prediction with linear regression model."""
    return tx @ w

def f1_score(y_targ, y_pred):
    """Compute the F1 score of a prediction."""
    mask_targ = (y_targ == 1)
    mask_pred = (y_pred == 1)

    # Total positives
    total_pred = np.count_nonzero(mask_pred)
    total_targ = np.count_nonzero(mask_targ)

    # True positives
    true_pos = np.count_nonzero(mask_pred[mask_targ])

    if (true_pos == 0) or (total_pred == 0) or (total_targ == 0):
        return 0.0

    precision = true_pos / total_pred
    recall = true_pos / total_targ

    # Compute F1 score
    score = 2*(precision*recall)/(precision+recall)

    return score

def accuracy(y_targ, y_pred):
    """Compute the accuracy of a prediction"""
    total_wrong = np.count_nonzero(y_targ-y_pred)
    return 1.0 - (total_wrong / len(y_pred))
