import numpy as np
import helpers

def mse(y, tx, w):
    """
    Mean squared error loss function.
    
    Parameters
    ----------
    y : Vector
        Output.
    tx : Matrix
        Input.
    w : Vector
        weights.
    
    Returns
    -------
    loss function.
    
    """
    e = y - tx @ w
    return (1/(2*tx.shape[0])) * np.sum(e**2)

def mae(y, tx, w):
    """
    Mean absolute error.

    Parameters
    ----------
    y : Vector
        Output.
    tx : Array
        Input.
    w : Vector
        Weights.

    Returns
    -------
    loss function.

    """
    
    return np.mean(np.abs(y - tx @ w))

def logistic_error(y, tx, w):
    """
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    tx : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    a = helpers.sigmoid( y - tx @ w)
    loss = (- 1 / tx.shape[0]) * np.sum(y * np.log(a) + (1 - y) * (np.log(1 - a)))
    return loss