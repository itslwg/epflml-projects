
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

def split_data(x, y, ratio, seed=1, shuffle=False):
    """
    Split data into train and test set

    Parameters
    ----------
    x : Matrix
        Input features.
    y : Vector
        output.
    ratio : Scalar
        What proportion of the data to keep for train vs test.  E.g, 0.8 --> 80% of data is the train set & 20% test set.
    seed : Scalar, optional
        Random state. The default is 1.
    shuffle : Boolean, optional
        Shuffle the data or not. The default is False.

    Returns
    -------
    x_train : Matrix
        Training set input features.
    x_test : Matrix
        Testing set input features.
    y_train : Vector
        Training set output.
    y_test : Vector
        testing set output.

    """
    assert len(x) == len(y), "X & y must be the same length"
    # set seed
    np.random.seed(seed)
    
    if shuffle:
        # Shuffle the data
        np.random.shuffle(x)
        np.random.shuffle(y)
    
    # Select what index we use to make the split
    
    split = int(x.shape[0]*ratio)
    
    # Make the split
    
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]
    
    return x_train, x_test, y_train, y_test
