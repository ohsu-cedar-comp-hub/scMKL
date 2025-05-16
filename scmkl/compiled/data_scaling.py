import numpy as np
from numba import njit


@njit
def center_data(x : np.ndarray, x_means : np.ndarray, sds : np.ndarray):
    '''
    Takes the training and testing data and calculates z-scores for 
    all elements in both train and test. Should not be used for 
    binary data.

    Parameters
    ----------
    x : np.ndarray
        > A 2D array of data for training as samples x features.

    means : np.ndarray
        > A 1D array of means corresponding to the columns in x.

    sds : np.ndarray
        > A 1D array of standard deviations corresponding to columns 
        in x_train and x_test.

    Returns
    -------
    x : np.array
        > A 2D array of z-scores calculated for each element.
    '''
    # Transforming x values to z-scores
    z_scores = (x - x_means) / sds

    return z_scores