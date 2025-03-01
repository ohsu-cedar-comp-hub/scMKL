import numpy as np
import scipy

from scmkl.calculate_z import _process_data


def estimate_sigma(adata, n_features = 5000):
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

    # Loop over every group in group_dict
    for group_features in adata.uns['group_dict'].values():

        # Select only features in that group and downsample for scalability
        num_group_features = len(group_features)
        group_array = np.array(list(group_features))
        n_feats = min([n_features, num_group_features])
        group_features = adata.uns['seed_obj'].choice(group_array, n_feats, 
                                                      replace = False) 

        # Use on the train data to estimate sigma
        X_train = adata[adata.uns['train_indices'], group_features].X
        X_train = _process_data(X_train = X_train, 
                                scale_data = adata.uns['scale_data'], 
                                return_dense = True)
        
        # Sample cells for scalability
        sample_idx = np.arange(X_train.shape[0])
        n_samples = np.min((2000, X_train.shape[0]))
        distance_indices = adata.uns['seed_obj'].choice(sample_idx, n_samples)

        # Calculate Distance Matrix with specified metric
        sigma = scipy.spatial.distance.cdist(X_train[distance_indices,:], 
                                             X_train[distance_indices,:], 
                                             adata.uns['distance_metric'])
        sigma = np.mean(sigma)

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