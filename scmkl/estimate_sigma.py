import scipy
import anndata
import numpy as np

from scmkl.data_processing import *
from scmkl.tfidf_normalize import _tfidf


def batch_sigma(X_train, adata, batches, batch_size) -> float:
    '''
    Calculates the kernel width (sigma) for a feature grouping through 
    sample batching.

    Parameters
    ----------
    X_train : np.ndarray
        > A 2D numpy array with cells x features with features filtered 
        to features in grouping and sampled cells.

    adata : anndata.AnnData
        > adata used to derive X_train containing 'seed_obj' in uns 
        attribute.

    batches : int
        > Number of minibatches to calculate sigma for.

    batch_size : int
        > Number of cells in each batch.

    Returns 
    -------
    sigma : float
        > The estimated group kernel with for Z projection before 
        adjustments for small kernel width or large groupings.
    '''
    # Calculate Distance Matrix with specified metric
    batch_sigmas = np.zeros(batches)

    for i in np.arange(batches):
        batch_indices = adata.uns['seed_obj'].choice(np.arange(X_train.shape[0]), 
                                                     batch_size, replace = False)
        batch_sigma = scipy.spatial.distance.pdist(X_train[batch_indices, :], 
                                                   metric = adata.uns['distance_metric'])
        batch_sigmas[i] = np.mean(batch_sigma)

    sigma = np.mean(batch_sigmas)

    return sigma


def est_group_sigma(adata, X_train, n_group_features, n_features, batches, 
                    batch_size) -> float:
    '''
    Processes data and calculates the kernel width (sigma) for a 
    feature grouping through sample batching.

    Parameters
    ----------
    X_train : np.ndarray
        > A 2D numpy array with cells x features with features filtered 
        to features in grouping and sampled cells.

    adata : anndata.AnnData
        > adata used to derive X_train containing 'seed_obj' in uns 
        attribute.

    n_group_features : int
        > Number of features in feature grouping.

    n_features : int
        > Maximum number of features to be used in sigma estimation.

    batches : int
        > Number of minibatches to calculate sigma for.

    batch_size : int
        > Number of cells in each batch.

    Returns 
    -------
    sigma : float
        > The estimated group kernel with for Z projection.
    '''    
    X_train = process_data(X_train, 
                            scale_data = adata.uns['scale_data'], 
                            return_dense = True)
    

    if adata.uns['tfidf']:
        X_train = _tfidf(X_train, mode = 'normalize')

    if adata.uns['reduction'].lower() == 'svd':
        X_train = svd_transformation(X_train)

    elif adata.uns['reduction'].lower() == 'pca':
        X_train = pca_transformation(X_train)

    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_train = np.array(X_train, dtype = np.float32)

    # Calculates mean sigma from all batches
    sigma = batch_sigma(X_train, adata, batches, batch_size)

    # sigma = 0 is numerically unusable in later steps
    # Using such a small sigma will result in wide distribution, and 
    # typically a non-predictive Z
    if sigma == 0:
        sigma += 1e-5

    if n_features < n_group_features:
        # Heuristic we calculated to account for fewer features used in 
        # distance calculation
        sigma = sigma * n_group_features / n_features 

    return sigma


def estimate_sigma(adata, n_features = 5000, batches = 10, 
                   batch_size = 100) -> anndata.AnnData:
    '''
    Calculate kernel widths to inform distribution for projection of 
    Fourier Features. Calculates one sigma per group of features.

    Parameters
    ----------
    **adata** : *AnnData* 
        > Created by `create_adata`.
    
    **n_features** : *int*  
        > Number of random features to include when estimating sigma. 
        Will be scaled for the whole pathway set according to a 
        heuristic. Used for scalability.

    **batches**: *int*
        > The number of batches to use for the distance calculation.
        This will average the result of `batches` distance calculations
        of `batch_size` randomly sampled cells. More batches will converge
        to population distance values at the cost of scalability.

    **batch_size**: *int*
        > The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability.
        
    Returns
    -------
    **adata** : *AnnData*
        > Key added `adata.uns['sigma']` with grouping kernel widths.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata.uns['sigma']
    array([10.4640895 , 10.82011454,  6.16769438,  9.86156855, ...])
    '''
    sigma_list = []

    assert batch_size < len(adata.uns['train_indices']), ("Batch size much be "
                                                          "smaller than the "
                                                          "training set.")

    if batch_size > 2000:
        print("Warning: Batch sizes over 2000 may "
               "result in long run-time.")

    # Loop over every group in group_dict
    for m, group_features in enumerate(adata.uns['group_dict'].values()):

        n_group_features = len(group_features)
        n_samples = np.min((adata.uns['train_indices'].shape[0], 2000))

        # Filtering to only features in grouping and subsampling
        X_train = get_group_mat(adata, n_features = n_features, 
                            group_features = group_features, 
                            n_group_features = n_group_features,
                            n_samples = n_samples)

        # Estimating sigma
        sigma = est_group_sigma(adata, X_train, n_group_features, 
                                n_features, batches, batch_size)

        sigma_list.append(sigma)
    
    adata.uns['sigma'] = np.array(sigma_list)
        
    return adata