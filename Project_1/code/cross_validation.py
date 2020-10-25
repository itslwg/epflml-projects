import numpy as np
import itertools as it
import time
import warnings
from implementations import *

warnings.filterwarnings('ignore')

def cross_validation(y_tr, tx_tr, y_te, tx_te, comb, verbose=2):
    # fit the model
    w, loss = reg_logistic_regression(
        y=y_tr,
        tx=tx_tr,
        initial_w=np.zeros((tx_tr.shape[1])),
        max_iters=1000,
        gamma=comb["gamma"],
        lambda_=comb["lambda_"],
        reg=comb["reg"],
        verbose=False,
        early_stopping=True,
        tol=0.001,
        patience=5
    )
    # calculate the loss for train and test data
    loss_tr = reg_logistic_error(y_tr, tx_tr, w, comb["lambda_"], comb["reg"])
    loss_te = reg_logistic_error(y_te, tx_te, w, comb["lambda_"], comb["reg"])
    # compute performance metrics
    p = logistic_prediction(tx_te, w)
    f1 = f1_score(y_te, p)
    acc = accuracy(y_te, p)

    return loss_tr, loss_te, f1, acc

def model_selection(y, tx, k_fold, degree, grid, seed, verbose=2):
    """Select the best model from all possible combinations of grid."""

    # Generate total permuations of hps for gridsearch
    k, v = zip(*grid.items())
    permutations = [dict(zip(k, values)) for values in it.product(*v)]
    losses = {
        "loss_tr": [],
        "loss_te": [],
        "f1_te": [],
        "acc_te": []
    }

    print(f'Training on {int((1-1/k_fold)*y.shape[0])} samples')
    for permutation in permutations:
        print(f'Seed: {seed}, Permutation: {str(permutation)}')
        y_trs, tx_trs, y_tes, tx_tes = prepare_split_data(y, tx, degree, k_fold, seed)
        # Cross validation
        trl = 0.0
        tel = 0.0
        f1 = 0.0
        acc = 0.0
        for k in range(k_fold):
            tr, te, f1l, accl = cross_validation(
                y_trs[k],
                tx_trs[k],
                y_tes[k],
                tx_tes[k],
                comb=permutation
            )
            trl += tr
            tel += te
            f1 += f1l
            acc += accl

            if verbose > 1: print(f'\t\t\tTR={tr:.4e} | TE={te:.4e} | F1:{f1l:.4f} | ACC:{accl:.4f}')

            if ~np.all(np.isfinite((tr, te, f1, acc))): break

        # k+1 needed to account for early exit due to NaNs or infs.
        losses["loss_tr"].append(trl / (k+1))
        losses["loss_te"].append(tel / (k+1))
        losses["f1_te"].append(f1 / (k+1))
        losses["acc_te"].append(acc / (k+1))

        if verbose > 0: print('\t'+' | '.join([ f'{key} = {val[-1]:.4f}' for key, val in losses.items() ]))

    idx = np.argmax(losses["acc_te"])
    print('\nMINIMUMS:')
    print(f'\tBest parameters: {str(permutations[idx])}')
    print(f'\tAccuracy: {losses["acc_te"][idx]:.4f}')
    print(f'\tF1-score: {losses["f1_te"][idx]:.4f}')
    print(f'\tLoss: {losses["loss_te"][idx]:.4e}')

    return permutations[idx]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def prepare_split_data(y, tx, degree, k_fold, seed):
    """
    Split the dataset based on k-fold cross validation and prepare features.
    Returns the k splits.
    """
    k_indices = build_k_indices(y, k_fold, seed)

    tx_trs = []
    y_trs = []
    tx_tes = []
    y_tes = []

    for k in range(k_fold):
        # get k-th subgroup in test, others in train
        mask = np.full((y.shape[0]), False, dtype=bool)
        mask[k_indices[k]] = True

        y_te = y[mask]
        tx_te = tx[mask]

        mask = ~mask
        y_tr = y[mask]
        tx_tr = tx[mask]

        # Replace NaNs and standarize
        tx_tr, mean, std, mean_nan, _ = prepare_features(tx_tr, degree)
        tx_te, _, _, _, _ = prepare_features(tx_te, degree, mean_nan, mean, std)

        tx_trs.append(tx_tr)
        y_trs.append(y_tr)
        tx_tes.append(tx_te)
        y_tes.append(y_te)

    return y_trs, tx_trs, y_tes, tx_tes

if __name__ == "__main__":

    # Import data
    print("... Importing data ..."); start = time.time()

    train, test, cols = import_data(path = "data/")

    print(f"Done in {time.time() - start:.2f} seconds.")

    # Split X and y
    tx_tr, y_tr, tx_te = split_X_y(train, test, cols)

    # Grid search model selection
    grid = {
        "gamma": np.logspace(-1, 0, 10),
        "lambda_": np.logspace(-3, 0, 10),
        "reg": [2]
    }
    seed = 42
    reg = 2 # L2 regularization
    k_fold = 4
    verbose = 2
    degree = 3

    print("... Starting model selection ..."); start = time.time()
    params = model_selection(
        y=y_tr,
        tx=tx_tr,
        k_fold=k_fold,
        degree=degree,
        grid=grid,
        seed=seed,
        verbose=verbose
    )

    print(f"Done in {time.time() - start:.2f} seconds.")

    # Save results to file
    with open("data/hyper_parameters.txt", "w") as f:
        for key in params.keys():
            f.write(f"{key},{params[key]}\n")
