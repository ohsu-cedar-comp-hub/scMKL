import numpy as np
import numba as nb        


@nb.njit(fastmath = True, cache = True)
def gaussian_trans(X: np.ndarray, sigma: float, 
                   seed_obj: np.random._generator.Generator,
                   D: int):
    """
    Function to sample the projection direction vector W
    for calculating gaussian Random Fourier Features (RFF)
    for X.

    Parameters
    ----------
    X : np.ndarray
        Data matrix X to calculate an approximate kernel of.
    
    sigma : float
        Parameter from data distribution controlling the approximate 
        kernel width.

    seed_obj : np.random._generator.Generator
        Numpy random generator object from `adata.uns['seed_obj']`.

    D : int
        Parameter determining the number of RFF used to approximate 
        the kernel function.

    Returns
    -------
    W : np.ndarray
        Vector defining the direction of the projection of RFF.
    """
    gamma = 1 / ( 2*sigma**2)
    sigma_p = 0.5*np.sqrt(2*gamma)

    W = seed_obj.normal(0, sigma_p, X.shape[1]*D)
    W = W.reshape((X.shape[1]), D)

    return W


@nb.njit(fastmath = True, cache = True)
def laplacian_trans(X: np.ndarray, sigma: float, seed_obj, d: int):
    """
    Function to sample the projection direction vector W
    for calculating laplacian Random Fourier Features (RFF)
    for X.

    Parameters
    ----------
    X : np.ndarray
        Data matrix X to calculate an approximate kernel of.
    
    sigma : float
        Parameter from data distribution controlling the approximate 
        kernel width.

    seed_obj : np.random._generator.Generator
        Numpy random generator object from `adata.uns['seed_obj']`.

    D : int
        Parameter determining the number of RFF used to approximate 
        the kernel function.

    Returns
    -------
    W : np.ndarray
        Vector defining the direction of the projection of RFF.
    """
    gamma = 1 / (2 * sigma)

    W = seed_obj.standard_cauchy(X.shape[1] * d)
    W = gamma * W.reshape((X.shape[1], d))

    return W


@nb.njit(fastmath = True, cache = True)
def cauchy_trans(X: np.ndarray, sigma: float, seed_obj, d: int):
    """
    Function to sample the projection direction vector W
    for calculating cauchy Random Fourier Features (RFF)
    for X.

    Parameters
    ----------
    X : np.ndarray
        Data matrix X to calculate an approximate 
        kernel of.
    
    sigma : float
        Parameter from data distribution controlling the approximate 
        kernel width.

    seed_obj : np.random._generator.Generator
        Numpy random generator object from `adata.uns['seed_obj']`.

    D : int
        Parameter determining the number of RFF used to approximate 
        the kernel function.

    Returns
    -------
    W : np.ndarray
        Vector defining the direction of the projection of RFF.
    """
    gamma = 1 / (2 * sigma ** 2)
    b = 0.5 * np.sqrt(gamma)

    W = seed_obj.laplace(0, b, X.shape[1] * d)
    W = W.reshape((X.shape[1], d))

    return W
