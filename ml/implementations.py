# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import template.costs


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i, w0_star in enumerate(w0):
        for j, w1_star in enumerate(w1):
            losses[i, j] = compute_loss(y, tx, np.array([w0_star, w1_star]))

    return losses

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y[np.newaxis].T - np.dot(tx, w[np.newaxis].T)
    return - ((1 / tx.shape[0]) * np.dot(tx.T, e)).T


def gradient_descent(y, tx, initial_w, max_iters, gamma):
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
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_loss(y, tx, w):
    """Calculate the MSE loss."""
    e = y[np.newaxis].transpose() - tx.dot(w[np.newaxis].T)
    mse = (1 / (2*tx.shape[0])) * np.sum(np.square(e))
    
    return mse
