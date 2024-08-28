import numpy as np
import scipy

from scmkl.data_processing import process_data


def estimate_sigma(adata, n_features = 5000):
    '''
    Function to calculate approximate kernels widths to inform distribution for project of Fourier Features. Calculates one sigma per group of features
    Input:
            adata- Adata obj as created by `Create_Adata`
            n_features- Number of random features to include when estimating sigma.  Will be scaled for the whole pathway set according to a heuristic. Used for scalability
    Output:
            adata object with sigma values.  Sigmas accessible by adata.uns['sigma']

    '''
 
    sigma_list = []

    # Loop over every group in group_dict
    for group_name, group_features in adata.uns['group_dict'].items():

        # Select only features within that group and downsample for scalability
        num_group_features = len(group_features)
        group_features = adata.uns['seed_obj'].choice(np.array(list(group_features)), min([n_features, num_group_features]), replace = False) 

        X_train = adata[adata.uns['train_indices'], group_features].X

        X_train = process_data(X_train = X_train, data_type = adata.uns['data_type'], return_dense = True)
        
        # Sample cells because distance calculation are costly and can be approximated
        distance_indices = adata.uns['seed_obj'].choice(np.arange(X_train.shape[0]), np.min((2000, X_train.shape[0])))

        # Calculate Distance Matrix with specified metric
        sigma = np.mean(scipy.spatial.distance.cdist(X_train[distance_indices,:], X_train[distance_indices,:], adata.uns['distance_metric']))

        if sigma == 0:
            sigma += 1e-5

        if n_features < num_group_features:
            sigma = sigma * num_group_features / n_features # Heuristic we calculated to account for fewer features used in distance calculation

        sigma_list.append(sigma)
    
    adata.uns['sigma'] = np.array(sigma_list)
        
    return adata