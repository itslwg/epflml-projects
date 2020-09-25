# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y[np.newaxis].transpose() - tx.dot(w[np.newaxis].T)
    mse = (1 / (2*tx.shape[0])) * np.sum(np.square(e))
    
    return mse
