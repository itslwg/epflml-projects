
import numpy as np

def f1_score(y_targ, y_pred):
    """
    Compute F1 score of a prediction

    Parameters
    ----------
    y_targ : Numpy array
        Target labels. y_targ[i] must be 0 (background) or 1 (signal).
    y_pred: Numpy array
        Predicted labels. y_pred[i] must be 0 (background) or 1 (signal).

    Returns
    -------
    score : Real scalar
        the F1 score of the prediction.
    """

    mask_targ = (y_targ == 1)
    mask_pred = (y_pred == 1)

    # Total positives
    total_pred = np.count_nonzero(mask_pred)
    total_targ = np.count_nonzero(mask_targ)

    # True positives
    true_pos = np.count_nonzero(mask_pred[mask_targ])

    precision = true_pos / total_pred
    recall = true_pos / total_targ

    # Compute F1 score
    score = 2*(precision*recall)/(precision+recall)

    return score


def accuracy(y_targ, y_pred):
    """
    Compute accuracy of a prediction

    Parameters
    ----------
    y_targ : Numpy array
        Target labels. y_targ[i] must be 0 (background) or 1 (signal).
    y_pred: Numpy array
        Predicted labels. y_pred[i] must be 0 (background) or 1 (signal).

    Returns
    -------
    score : Real scalar
        the accuracy of the prediction.
    """

    total_wrong = np.count_nonzero(y_targ-y_pred)

    return 1.0 - (total_wrong / len(y_pred))


def r_squared(y_targ, y_pred):
    """
    Compute the coefficient of determination

    Parameters
    ----------
    y_targ : Vector
        True output.
    y_pred : Vector
        Predicted output.

    Returns
    -------
    rs : Scalar
        r squared.

    """
    
    ss_tot = np.sum((y_targ - np.mean(y_targ))**2)
    ss_res = np.sum((y_targ - y_pred)**2)
    return 1 - (ss_res/ss_tot)

