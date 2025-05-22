import numpy as np
import numba as nb


@nb.njit(cache = True, fastmath = True)
def diff(sample_1, sample_2):
    '''
    Calculates the difference between all features of two samples.
    '''
    result = sample_1 - sample_2

    return result


@nb.njit(cache = True, fastmath = True)
def sq_diff(sample_1 : np.ndarray, sample_2 : np.ndarray):
    '''
    Calculates the squared difference between all features of two 
    samples.
    '''
    # Calc difference between all features for samples
    result = sample_1 - sample_2

    # For small values, mult operator is faster
    result *= result

    return result


@nb.njit(cache = True, fastmath = True)
def euclidean_distance(sample_1, sample_2):
    '''
    Takes a vector of the squared difference between features and 
    returns the euclidean distance between the two samples.
    '''
    # Calculating squared differences
    differences = sq_diff(sample_1, sample_2)

    value = np.sum(differences)
    value = np.sqrt(value)

    return value


@nb.njit(cache = True, fastmath = True)
def cityblock_distance(sample_1, sample_2):
    '''
    
    '''
    differences = diff(sample_1, sample_2)

    value = np.abs(differences)
    value = np.sum(value)

    return value


@nb.njit(cache = True, parallel = True, fastmath = True)
def cdist(x : np.ndarray, dist_calc):
    '''
    Calculates the euclidean distance matrix.
    '''
    # Creating array to capture distances
    dist = np.zeros((x.shape[0], x.shape[0]))

    for i in nb.prange(x.shape[0]):
        min_val = min(i, x.shape[0])

        # Prevents making the same comparison again and 
        # comparing samples to themselves
        for j in range(min_val, x.shape[0]):
            result = dist_calc(x[i,:], x[j,:])
            dist[i,j] = result
            dist[j,i] = result

    return dist