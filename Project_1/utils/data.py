
""" Will be used in the run.py script that takes the test file and generates the predictions. No pandas allowed"""
import numpy as np

def import_data(path):
    #TODO
    """
    Import csv file and return array of the data and vector of the column names.

    Parameters
    ----------
    path : str
        Directory of csv file.

    Returns
    -------
    data : np.array
        Array of dataset.
    cols : np.array
        Name of columns.

    """
    pass
    #return data, cols
    
def clean(data, cols, col_med=None, to_drop=None):
    """
    When used on the training data:
        - Turn string into numeric
        - Replace -999 into nan
        - Delete columns where more than 50% of rows are missing
        - Replace nan values to median of the column
        - Split into X and y
    When used on the testing  data:
        - Turn string into numeric
        - Replace -999 into nan
        - Delete same columns as for training set (not based on 50% missing)
        - Replace nan values with median of the column OF THE TRAINING SET.
        - Split into X and y.

    Parameters
    ----------
    data : np.array
        dataset.
    cols : np.array
        Column names.
    col_med : np.array, optional
        Median for each column we keep from the tr set. The default is None.
    to_drop : np.array, optional
        Names of columns to drop. The default is None.

    Returns
    -------
    X : np.array
        Features.
    y : np.array
        Target.

    """
    
    # Turn string column (target) into numeric value
    pred_idx = np.argwhere(cols=="Prediction")[0,0]
    preds = data[:,pred_idx].copy()
    preds = np.where(preds=="s", 1, 0)
    data[:,pred_idx] = preds
    
    # Replace -999 into nan
    data[data == -999] = np.nan
    
    # Change fmt so isnan() works
    # TODO: When we import w/o pandas, we will not need to do this anymore.
    data = data.astype("float64")
    
    # Remove columns where there is more than 50% of rows missing
    if not to_drop:
        count_nan = np.sum(np.isnan(data), axis=0)
        to_keep_idx = np.where(count_nan >= int(0.5*data.shape[0]), False, True)
        data = data[:,to_keep_idx]
    else:
        # TODO: Remove list of columns that has been specified.
        pass
    
    # Replace remaining nan values with the median of the column
    if not col_med:   
        col_med = np.nanmedian(data, axis=0)
        
    repl_idx = np.where(np.isnan(data))
    data[repl_idx] = np.take(col_med, repl_idx[1])
    
    # Split features and target   
    id_idx = np.argwhere(cols=="Id")[0,0]
    x_idx = np.delete(np.arange(data.shape[1]), [id_idx, pred_idx])
    y = data[:,pred_idx]
    X = data[:,x_idx]
    
    return X, y