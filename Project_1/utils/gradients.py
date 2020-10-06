
import numpy as np
from helpers import *

def mse_grad(y, tx, w):
    """
    Compute gradient for mse loss

    Parameters
    ----------
    y : Vector
        Input.
    tx : Matrix
        Output.
    w : Vector
        Weights.

    Returns
    -------
    Vector of gradients.

    """
    e = y - tx @ w
    return (-1/tx.shape[0]) * tx.T @ e

def mae_grad(y, tx, w):
    """
    Compute subgradient for mae loss

    Parameters
    ----------
    y : Vector
        Input.
    tx : Matrix
        Output.
    w : Vector
        Weights.

    Returns
    -------
    Vector of gradients.

    """
    
    e = y - tx @ w
    return (-1/tx.shape[0]) * tx.T @ np.sign(e)

def logistic_grad(y, tx, w):
    """
    Compute gradient for logistic loss function

    Parameters
    ----------
    y : Vector
        Input.
    tx : Matrix
        Output.
    w : Vector
        Weights.

    Returns
    -------
    Vector of gradients.

    """
    
    e = sigmoid(tx @ w) - y
    return (1/tx.shape[0]) * tx.T @ e


def reg_logistic_grad(y, tx, w, lambda_, reg):
    """
    Compute gradient for logistic loss function with regularization. 

    Parameters
    ----------
    y : Vector
        Output.
    tx : Matrix
        Input.
    w : Vector
        Weights.
    lambda_ : Scalar
        Constant.
    reg : Scalar
        L1 or L2 regularization.

    Returns
    -------
    grad : Vector
        Gradient wrt to the weights.
    """
    assert (reg==1 or reg==2), "reg needs to be 1 or 2"
    
    if (reg==1):
        # L1 regularization
        return logistic_grad(y, tx, w) + lambda_ * np.sign(w)
    else:
        # L2 regularization 
        return logistic_grad(y, tx, w) + 2 * lambda_ * w