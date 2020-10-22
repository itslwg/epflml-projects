
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
        w = w.ravel()
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
    verbose : Boolean, optional
        Print GD steps. The default is False.

    Returns
    -------
    losses : Vector
        Loss for each iteration.
    ws : Vector
        Weight vector for each iteration.

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
        w = w.ravel()
        # Store the weights and loss
        ws.append(w)
        losses.append(loss)
        
        if verbose:
            print("Gradient descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws       
    
def reg_gradient_descent_logistic(y, tx, initial_w, max_iters, gamma, lambda_, reg, verbose=False,
                                  early_stopping=True, tol=0.0001, patience=5):
    """
    Regularized logistic regression with gradient descent.

    Parameters
    ----------
    y : Vector
        Output.
    tx : Matrix
        Input.
    initial_w : Vector
        Inital weights.
    max_iters : Scalar
        Maximum iterations.
    gamma : Scalar
        Learning rate.
    lambda_ : Scalar
        Regularization.
    reg : Integer
        l1 or l2 regularizarion.
    verbose : Boolean, optional
        Print steps. The default is False.
    early_stopping : Boolean, optional
        Stop early if no improvement in loss. The default is True.
    tol : Scalar, optional
        Minimum the loss needs to decrease by. The default is 0.0001.
    patience : Integer, optional
        Change needs to be seen in this number of iterations. The default is 10.

    Returns
    -------
    losses : Vector
        History of losses.
    ws : Vector
        History of weights.

    """

    # Define parameters to store w and loss.
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
            print("Gradient descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
        if (early_stopping) and (n_iter > patience):
            # Check if loss has improved by tol in last patience iters
            l_pat = reg_logistic_error(y, tx, ws[-patience], lambda_, reg)
            l_1 = reg_logistic_error(y, tx, ws[-1], lambda_, reg)
            try: # If we get invaluid value in the log.
                if ((l_pat - l_1) < tol):
                    print(f"Stopped after {n_iter} it.")
                    break
            except exception as e:
                print(e)
                break
        
    return losses, ws