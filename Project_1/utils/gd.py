
import numpy as np
from helpers import *
from costs import *
from gradients import *

def gradient_descent(y, tx, initial_w, max_iters, gamma, verbose = False):
    """
    Gradient descent algorithm.    

    Parameters
    ----------
    y : Vector
        Output.
    tx : Matrix
        Input.
    initial_w : Vector
        Initial weights.
    max_iters : Scalar
        Maximum number of iterations.
    gamma : Scalar
        Learning rate.
    verbose : TYPE, optional
        Print GD steps. The default is False.

    Returns
    -------
    vector of loss and vector of weights for each iteration.

    """
    # Define parameters to store w and loss.
    ws = [initial_w]
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters):
        # Compute gradient and loss
        g = mse_grad(y, tx, w)
        loss = mse(y, tx, w)
        # Update the weights
        w = w - gamma * g
        # Store the weights and loss
        ws.append(w)
        losses.append(loss)
        
        if verbose:
            print("Gradient descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws
        
def gradient_descent_logistic(y, tx, initial_w, max_iters, gamma, verbose = False):
    """
    Gradient descent algorithm for logistic regression.    

    Parameters
    ----------
    y : Vector
        Output.
    tx : Matrix
        Input.
    initial_w : Vector
        Initial weights.
    max_iters : Scalar
        Maximum number of iterations.
    gamma : Scalar
        Learning rate.
    verbose : TYPE, optional
        Print GD steps. The default is False.

    Returns
    -------
    vector of loss and vector of weights for each iteration.

    """
    # Define parameters to store w and loss.
    ws = [initial_w]
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters):
        # Compute gradient and loss
        g = logistic_grad(y, tx, w)
        loss = logistic_error(y, tx, w)
        # Update the weights
        w = w - gamma * g
        # Store the weights and loss
        ws.append(w)
        losses.append(loss)
        
        if verbose:
            print("Gradient descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws       
    
