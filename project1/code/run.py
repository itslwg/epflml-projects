import numpy as np
import time
import warnings
from implementations import *

warnings.filterwarnings('ignore')

# Parameters
degree = 3
gamma = 0.59948
lambda_ = 0.001
reg = 2
max_iters = 1000 # Can be change to speed up training
tol = 0.0001 # Can be change to speed up training
patience = 20 # Can be change to speed up training

# Import data
print("... Importing data ..."); start = time.time()

train, test, cols = import_data(path = "data/")

print(f"Done in {time.time() - start:.2f} seconds.")

# Prepare train and test data
print("... Preparing data for learning ...."); start = time.time()

tx_tr, y_tr, tx_te = split_X_y(train, test, cols)

tx_tr, mean, std, mean_nan, nan_cols = prepare_features(tx_tr, degree)
tx_te, _, _, _, _ = prepare_features(tx_te, degree, mean_nan, mean, std)

print(f"Done in {time.time() - start:.2f} seconds.")

# Train model
print("... Training model ..."); start = time.time()

initial_w = np.zeros((tx_tr.shape[1]))
w, loss = reg_logistic_regression(
        y=y_tr,
        tx=tx_tr,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        lambda_=lambda_,
        reg=reg,
        verbose=True,
        early_stopping=True,
        tol=tol,
        patience=patience
    )

print(f"Done in {time.time() - start:.2f} seconds.")

# Compute in-sample metrics (sanity check).
y_pred = logistic_prediction(tx_tr, w)
f1 = f1_score(y_tr, y_pred)
acc = accuracy(y_tr, y_pred)

print('In-sample results:')
print(f'\tLoss:     {loss:.4e}')
print(f'\tF1 score: {f1:.4f}')
print(f'\tAccuracy: {acc:.4f}')

# Generate submission
ids = test[:,0]
oos_pred = logistic_prediction(tx_te, w)
oos_pred[oos_pred == 0] = -1
oos_pred[oos_pred == 1] = 1

create_csv_submission(ids, oos_pred, 'data/submission.csv')
