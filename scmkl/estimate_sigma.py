import scipy
import anndata as ad
import numpy as np

from scmkl.data_processing import process_data, get_group_mat, sample_cells
from scmkl.tfidf_normalize import tfidf


def get_batches(sample_range: list | np.ndarray, 
                seed_obj: np.random._generator.Generator, 
                batches: int,
                batch_size: int) -> np.ndarray:
    """
    Gets batch indices for estimating sigma.

    Parameters
    ----------
    sample_range : list | np.ndarray
        A 1D array with first element being 0 and last element being 
        (1 - number of samples from X_train).

    seed_obj : np.random._generator.Generator
        Numpy random generator object from `adata.uns['seed_obj']`.

    batches : int
        Number of batches to calculate indices for.

    batch_size : int
        Number of samples in each batch.

    Returns
    -------
    batches_idx : np.ndarray
        A 2D array with each row cooresponding to the sample indices 
        for each batch.
    """
    required_n = batches*batch_size
    train_n = len(sample_range)
    assert required_n <= train_n, (f"{required_n} cells required for "
                                   f"{batches} batches of {batch_size} cells; "
                                   f"only {train_n} cells present")

    # Creating a mat of batch x sample indices for estimating sigma
    batches_idx = np.zeros((batches, batch_size), dtype = np.int16)

    for i in range(batches):
        batches_idx[i] = seed_obj.choice(sample_range, 
                                         batch_size, 
                                         replace = False)
        
        # Removing selected indices from sample options
        rm_indices = np.isin(sample_range, batches_idx[i])
        sample_range = np.delete(sample_range, rm_indices)

    return batches_idx


def batch_sigma(X_train: np.ndarray,
                distance_metric: str,
                batch_idx: np.ndarray) -> float:
    """
    Calculates the kernel width (sigma) for a feature grouping through 
    sample batching.

    Parameters
    ----------
    X_train : np.ndarray
        A 2D numpy array with cells x features with features filtered 
        to features in grouping and sampled cells.

    distance_metric: str
        The pairwise distance metric used to estimate sigma. Must
        be one of the options used in scipy.spatial.distance.cdist.

    batch_idx: np.ndarray
        A 2D array with each row cooresponding to the sample indices 
        for each batch.    

    Returns
    -------
    sigma : float
        The estimated group kernel with for Z projection before 
        adjustments for small kernel width or large groupings.
    """
    # Calculate Distance Matrix with specified metric
    n_batches = batch_idx.shape[0]
    batch_sigmas = np.zeros(n_batches)

    for i in np.arange(n_batches):
        idx = batch_idx[i]
        batch_sigma = scipy.spatial.distance.pdist(X_train[idx,:], 
                                                   distance_metric)
        batch_sigmas[i] = np.mean(batch_sigma)

    sigma = np.mean(batch_sigmas)

    return sigma


def est_group_sigma(adata: ad.AnnData,
                    X_train: np.ndarray,
                    n_group_features: int,
                    n_features: int, 
                    batch_idx: np.ndarray) -> float:
    """
    Processes data and calculates the kernel width (sigma) for a 
    feature grouping through sample batching.

    Parameters
    ----------
    X_train : np.ndarray
        A 2D numpy array with cells x features with features filtered 
        to features in grouping and sampled cells.

    adata : anndata.AnnData
        adata used to derive X_train containing 'seed_obj' in uns 
        attribute.

    n_group_features : int
        Number of features in feature grouping.

    n_features : int
        Maximum number of features to be used in sigma estimation.

    batch_idx
        A 2D array with each row cooresponding to the sample indices 
        for each batch.

    Returns
    -------
    sigma : float
        The estimated group kernel with for Z projection.

    """   
    if adata.uns['tfidf']:
        X_train = tfidf(X_train, mode = 'normalize')

    X_train = process_data(X_train, 
                        scale_data = adata.uns['scale_data'], 
                        return_dense = True)

    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_train = np.array(X_train, dtype = np.float32)

    # Calculates mean sigma from all batches
    sigma = batch_sigma(X_train, adata.uns['distance_metric'], batch_idx)

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


def estimate_sigma(adata: ad.AnnData,
                   n_features: int = 5000,
                   batches: int = 10, 
                   batch_size: int = 100) -> ad.AnnData:
    """
    Calculate kernel widths to inform distribution for projection of 
    Fourier Features. Calculates one sigma per group of features.

    Parameters
    ----------
    adata : ad.AnnData
        Created by `create_adata`.
    
    n_features : int
        Number of random features to include when estimating sigma. 
        Will be scaled for the whole pathway set according to a 
        heuristic. Used for scalability.

    batches : int
        The number of batches to use for the distance calculation.
        This will average the result of `batches` distance calculations
        of `batch_size` randomly sampled cells. More batches will converge
        to population distance values at the cost of scalability.

    batch_size : int
        The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability.
        If `batches` * `batch_size` > # training cells,
        `batch_size` will be reduced to `int(num training cells / 
        batches)`.
        
    Returns
    -------
    adata : ad.AnnData
        Key added `adata.uns['sigma']` with grouping kernel widths.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata.uns['sigma']
    array([10.4640895 , 10.82011454,  6.16769438,  9.86156855, ...])
    """
    assert batch_size <= len(adata.uns['train_indices']), ("Batch size must be "
                                                          "smaller than the "
                                                          "training set.")

    if batch_size * batches > len(adata.uns['train_indices']):
        old_batch_size = batch_size
        batch_size = int(len(adata.uns['train_indices'])/batches)
        print("Specified batch size required too many cells for "
                "independent batches. Reduced batch size from "
                f"{old_batch_size} to {batch_size}")

    if batch_size > 2000:
        print("Warning: Batch sizes over 2000 may "
               "result in long run-time.")
        
    # Getting subsample indices
    sample_size = np.min((2000, adata.uns['train_indices'].shape[0]))
    indices = sample_cells(adata.uns['train_indices'], sample_size=sample_size, 
                           seed_obj=adata.uns['seed_obj'])

    # Getting batch indices
    sample_range = np.arange(sample_size)
    batch_idx = get_batches(sample_range, adata.uns['seed_obj'], 
                            batches, batch_size)

    # Loop over every group in group_dict
    sigma_array = np.zeros((len(adata.uns['group_dict'])))
    for m, group_features in enumerate(adata.uns['group_dict'].values()):

        n_group_features = len(group_features)

        # Filtering to only features in grouping using filtered view of adata
        X_train = get_group_mat(adata[indices], n_features=n_features, 
                            group_features=group_features, 
                            n_group_features=n_group_features)
        
        if adata.uns['tfidf']:
            X_train = tfidf(X_train, mode='normalize')

        # Data filtering, and transformation according to given data_type
        # Will remove low variance (< 1e5) features regardless of data_type
        # If scale_data will log scale and z-score the data
        X_train = process_data(X_train=X_train,
                               scale_data=adata.uns['scale_data'], 
                               return_dense=True)    

        # Estimating sigma
        sigma = est_group_sigma(adata, X_train, n_group_features, 
                                n_features, batch_idx=batch_idx)

        sigma_array[m] = sigma
    
    adata.uns['sigma'] = sigma_array
        
    return adata