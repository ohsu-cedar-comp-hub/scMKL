import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD, PCA

from scmkl.calculate_z import _process_data
from scmkl.tfidf_normalize import _tfidf


def estimate_sigma(adata, n_features = 5000, batches = 10, batch_size = 100):
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
        > Key added `adata.uns['sigma']`.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata.uns['sigma']
    array([10.4640895 , 10.82011454,  6.16769438,  9.86156855, ...])
    '''
    sigma_list = []

    assert batch_size < len(adata.uns['train_indices']),\
        'Batch size much be smaller than the training set.'

    if batch_size > 2000:
        print('Warning: Batch sizes over 2000 may \
               result in long run-time.')

    # Loop over every group in group_dict
    for m, group_features in enumerate(adata.uns['group_dict'].values()):

        # Select only features in that group and downsample for scalability
        num_group_features = len(group_features)
        group_array = np.array(list(group_features))
        n_feats = min([n_features, num_group_features])
        group_features = adata.uns['seed_obj'].choice(group_array, n_feats, 
                                                      replace = False) 

        # Use on the train data to estimate sigma
        X_train = adata[adata.uns['train_indices'], group_features].X

        
        # Sample cells for scalability
        sample_idx = np.arange(X_train.shape[0])
        n_samples = np.min((2000, X_train.shape[0]))
        distance_indices = adata.uns['seed_obj'].choice(sample_idx, n_samples, 
                                                        replace = False)

        X_train = _process_data(X_train = X_train, 
                                 scale_data = adata.uns['scale_data'], 
                                 return_dense = True)
        

        if adata.uns['tfidf']:
            X_train = _tfidf(X_train, mode = 'normalize')

        if adata.uns['reduction'].lower() == 'svd':

            SVD_func = TruncatedSVD(n_components = np.min([50, X_train.shape[1]]), random_state = 1)
            X_train = SVD_func.fit_transform(scipy.sparse.csr_array(X_train))[:,1:]

        elif adata.uns['reduction'].lower() == 'pca':
            PCA_func = PCA(n_components = np.min([50, X_train.shape[1]]), random_state = 1)
            X_train = PCA_func.fit_transform(np.asarray(X_train))

        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray()
            X_train = np.array(X_train, dtype = np.float32)

        # Calculating distances
        sigma = scipy.spatial.distance.pdist(X_train, adata.uns['distance_metric'])

        # Calculate Distance Matrix with specified metric
        batch_sigmas = np.zeros((batches))

        for i in np.arange(batches):
            batch_indices = adata.uns['seed_obj'].choice(np.arange(X_train.shape[0]), batch_size, replace = False)
            batch_sigma = scipy.spatial.distance.pdist(X_train[batch_indices, :], 
                                                       metric = adata.uns['distance_metric'])
            batch_sigmas[i] = np.mean(batch_sigma)

        sigma = np.mean(batch_sigmas)

        # sigma = 0 is numerically unusable in later steps
        # Using such a small sigma will result in wide distribution, and 
        # typically a non-predictive Z
        if sigma == 0:
            sigma += 1e-5

        if n_features < num_group_features:
            # Heuristic we calculated to account for fewer features used in 
            # distance calculation
            sigma = sigma * num_group_features / n_features 

        sigma_list.append(sigma)
    
    adata.uns['sigma'] = np.array(sigma_list)
        
    return adata