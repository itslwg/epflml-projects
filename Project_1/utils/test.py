#Script to check if the implementations work on random data.
#We do not give this to the teachers

import numpy as np
from helpers import *
from implementations import *

height, weight, gender = load_data(sub_sample=False, add_outlier=False)
x, mean_x, std_x = standardize(height)
y, tx = build_model_data(x, weight)

initial_w = np.array([0,0])
max_it = 50
gamma = 0.7

#  LR w/ GD
w, loss = least_squares_GD(y, tx, initial_w,
                           max_it, gamma, verbose=False)
print(f"LR w/ GD, w = {w}, loss = {loss}")

# LR w/ SGD --> I think there is an issue here
w, loss = least_squares_SGD(y, tx, initial_w,
                      max_it, gamma, verbose = False)
print(f"LR w/ SGD, w = {w}, loss = {loss}")

# LR w/ normal equations
w, loss = least_squares(y, tx)
print(f"LR w/ normal equations, w = {w}, loss = {loss}")

# Ridge regression w/ normal equations
# With lambda = 1.0, we get the same result as LR.
w, loss = ridge_regression(y, tx, lambda_ = 1.0)
print(f"Ridge regression w/ normal equations, w = {w}, loss = {loss}")

## Need a dataset to test logistic regression.