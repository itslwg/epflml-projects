import numpy as np
## Note: Set the working directory to the base of the project, i.e. the "Project_1" directory
from utils.helpers import *

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def compute_loss(y, tx, w, method = "MSE"):
    """Calculate the MSE or MAE loss."""
    
    if method == "MSE":
        e = y[np.newaxis].transpose() - tx.dot(w[np.newaxis].T)
        mse = (1 / (2*tx.shape[0])) * np.sum(np.square(e))
        
        return mse
    
    elif method == "MAE":
        
        return np.mean(np.abs(y-tx.dot(w)))
    
def sigmoid(x):
	"""Sigmoid function"""
	return 1 / (1 + np.exp(-x))

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i, w0_star in enumerate(w0):
        for j, w1_star in enumerate(w1):
            losses[i, j] = compute_loss(y, tx, np.array([w0_star, w1_star]))

    return losses


def compute_gradient(y, tx, w):
    """ Compute the gradient for MSE loss """
    e = y[np.newaxis].T - np.dot(tx, w[np.newaxis].T)
    return - ((1 / tx.shape[0]) * np.dot(tx.T, e)).T

def compute_gradient_logistic(y, tx, w):
    """ Compute the gradient for logistic loss function"""
    
    e = sigmoid(np.dot(tx, w[np.newaxis].T)) - y[np.newaxis].T
    return ((1 / tx.shape[0]) * np.dot(tx.T, e)).T

def compute_loss_logistic(y, tx, w):
    """ Logistic loss"""
    
    a = sigmoid(np.dot(tx, w[np.newaxis].T))
    loss = (- 1 / tx.shape[0]) * np.sum(y * np.log(a) + (1 - y) * (np.log(1 - a)))
    return loss

def gradient_descent(y, tx, initial_w, max_iters, gamma, verbose = False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        g = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # Update the weights
        w = w - gamma * g
        w = w.ravel()
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def gradient_descent_logistic(y, tx, initial_w, max_iters, gamma, verbose = False):
    """Gradient descent algorithm for logistic regression."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_gradient_logistic(y, tx, w)
        loss = compute_loss_logistic(y, tx, w)
        w-= gamma * g
        w = w.ravel()
        
        ws.append(w)
        losses.append(loss)
        
        if verbose:
            print(f"Gradient descent {n_iter}/{max_iters-1}: loss = {loss}, w0={w[0]}, w1={w[1]}")
    return losses, ws


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y[np.newaxis].T - np.dot(tx, w[np.newaxis].T)
    return - ((1 / tx.shape[0]) * np.dot(tx.T, e)).T


def stochastic_gradient_descent(y, tx, initial_w,
                                batch_size, max_iters, gamma,
                                verbose):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            g = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # Update the weights
            w = w - gamma * g
            w = w.ravel()
            # store w and loss
            ws.append(w)
            losses.append(loss)
            if verbose:
                print("Stochastic gradient descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares_GD(y, tx, initial_w,
                     max_it, gamma, verbose=False):
    """Linear Regression with Gradient Descent

    Uses Mean Squared Error as the loss function.
    """
    losses, ws = gradient_descent(
        y=y,
        tx=tx,
        initial_w=initial_w,
        max_iters=max_it,
        gamma=gamma,
        verbose=verbose
    )
    
    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w,
                      max_iters, gamma, verbose=False):
    """Linear regression with Stochastic Gradient Descent (SGD)

    Current implementation uses Mean Squared Error as the loss.
    """
    # Use batch_size = 1 as per the projec instructions.
    losses, ws = stochastic_gradient_descent(
        y=y,
        tx=tx,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        batch_size=1,
        verbose=verbose
    )
    return ws[-1], losses[-1]

def least_squares(y, tx):
    """Linear regression fit using normal equations."""
    # \hat{\beta} = (X^TX)^{-1} X^Ty
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w, compute_loss(y, tx, w)

def ridge_regression(y, tx, lambda_):
    """ Ridge regression fit using normal equations """
    # (X^T X + \lambda^{'}\mathbb{I})^{-1} X^T y
    n = tx.shape[1]
    w = np.linalg.inv( (tx.T @ tx) + (lambda_ * np.identity(n))) @ tx.T @ y
    return w, compute_loss(y, tx, w)



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression with gradient descent"""
    
    losses, ws = gradient_descent_logistic(
        y=y,
        tx=tx,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma
    )
    return ws[-1], losses[-1]
