import numpy as np
import implementations

from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# Linear regression

print("Linear Regession \n ---------------- \n")
X, y = datasets.load_boston(return_X_y = True)
X, _, _ = implementations.standardize_numpy(X)
tx = np.c_[np.ones(X.shape[0]), X]

initial_w = np.zeros(tx.shape[1])

max_iters = 1000
gamma = 0.01

w_lr, loss_lr = implementations.least_squares(y, tx)
y_pred_lr = tx @ w_lr
print(f"Linear regression eq: {r2_score(y_pred_lr, y)}")

w_lr_gd, loss_lr_gd = implementations.least_squares_GD(y, tx, initial_w,
                     max_iters, gamma, verbose=False)
y_pred_lr_gd = tx @ w_lr_gd
print(f"Linear regression gd: {r2_score(y_pred_lr_gd, y)}")

w_lr_sgd, loss_lr_sgd = implementations.least_squares_SGD(y, tx, initial_w,
                      max_iters, gamma, verbose=False)
y_pred_lr_sgd = tx @ w_lr_sgd
print(f"Linear regression sgd: {r2_score(y_pred_lr_sgd, y)}")

reg = LinearRegression().fit(X, y)
y_pred_sk = reg.predict(X)
print(f"Sklearn linear regression: {r2_score(y_pred_sk, y)}")


# Ridge regression

print("Ridge Reression \n -----------------")

lambda_ = 0.01

w_r, loss_r = implementations.ridge_regression(y, tx, lambda_)
y_pred_ridge = tx @ w_r
print(f"Ridge regression eq : {r2_score(y_pred_ridge, y)}")

y_pred_ridge_sk = Ridge(alpha=lambda_).fit(X, y).predict(X)
print("Sklearn ridge regression", r2_score(y_pred_ridge_sk, y))

# Logistic regression
print("Logistic Regression \n --------------")

X, y = datasets.load_breast_cancer(return_X_y = True)
X, _, _ = implementations.standardize_numpy(X)

tx = np.c_[np.ones(X.shape[0]), X]

initial_w = np.zeros(tx.shape[1])

w_log_gd, loss_log_gr = implementations.logistic_regression(y, tx, initial_w, max_iters,
                        gamma, verbose=False)
y_pred_log_gd = implementations.logistic_prediction(tx, w_log_gd)

w_log_gd_reg, loss_log_gd_reg = implementations.reg_logistic_regression(y, tx, lambda_, 2, initial_w,
                            max_iters, gamma, verbose=False,
                            early_stopping=True, tol = 0.0001,
                            patience = 5)
y_pred_log_gd_reg = implementations.logistic_prediction(tx, w_log_gd_reg)

print(f"Logistic regression gd : {implementations.accuracy(y, y_pred_log_gd)}")
print(f"Logistic regression reg: {implementations.accuracy(y, y_pred_log_gd_reg)}")

y_pred_log_sk = LogisticRegression().fit(X, y).predict(X)
print(f"Sklearn logistic regression : {implementations.accuracy(y, y_pred_log_sk)}")

y_pred_log_reg = LogisticRegression(C=1/lambda_, max_iter = 1000).fit(X,y).predict(X)
print(f"Sklearn reg logistic regression: {implementations.accuracy(y, y_pred_log_reg)}")



