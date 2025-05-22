import numpy as np
import numba as nb        


@nb.njit(fastmath = True, cache = True)
def gaussian_trans(x, adjusted_sigma, seed_obj, d):
    '''
    '''
    gamma = 1 / ( 2 * adjusted_sigma ** 2)
    sigma_p = 0.5 * np.sqrt(2 * gamma)

    w = seed_obj.normal(0, sigma_p, x.shape[1] * d)
    w = w.reshape((x.shape[1]), d)

    return w


@nb.njit(fastmath = True, cache = True)
def laplacian_trans(x, adjusted_sigma, seed_obj, d):
    '''
    
    '''
    gamma = 1 / (2 * adjusted_sigma)

    w = seed_obj.standard_cauchy(x.shape[1] * d)
    w = gamma * w.reshape((x.shape[1], d))

    return w


@nb.njit(fastmath = True, cache = True)
def cauchy_trans(x, adjusted_sigma, seed_obj, d):
    '''
    
    '''
    gamma = 1 / (2 * adjusted_sigma ** 2)
    b = 0.5 * np.sqrt(gamma)

    w = seed_obj.laplace(0, b, x.shape[1] * d)
    w = w.reshape((x.shape[1], d))

    return w

@nb.njit(fastmath = True, cache = True)
def add_projections(Z, x_idx, projection, sq_i_d):
    '''
    
    '''
    new_aks = np.hstack((np.cos(projection), np.sin(projection)))
    new_aks *= sq_i_d

    Z[0:, x_idx] = new_aks

    return Z
