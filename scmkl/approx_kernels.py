import numpy as np
import scipy

from scmkl.data_processing import process_data


def calculate_z(adata, n_features = 5000) -> tuple:
    '''
    Function to calculate approximate kernels.
    Input:
            adata- Adata obj as created by `Create_Adata`
                Sigma can be calculated with Estimate_Sigma or a heuristic but must be positive.
            n_features- Number of random feature to use when calculating Z- used for scalability
    Output:
            adata_object with Z matrices accessible with- adata.uns['Z_train'] and adata.uns['Z_test'] respectively

    '''
    assert np.all(adata.uns['sigma'] > 0), 'Sigma must be positive'

    #Number of groupings taking from group_dict
    N_pathway = len(adata.uns['group_dict'].keys())
    D = adata.uns['D']

    # Create Arrays to store concatenated group Z.  Each group of features will have a corresponding entry in each array
    Z_train = np.zeros((len(adata.uns['train_indices']), 2 * adata.uns['D'] * N_pathway))
    Z_test = np.zeros((len(adata.uns['test_indices']), 2 * adata.uns['D'] * N_pathway))

    # Loop over each of the groups and creating Z for each
    for m, group_features in enumerate(adata.uns['group_dict'].values()):
        
        #Extract features from mth group
        num_group_features = len(group_features)

        # group_feature_indices = adata.uns['seed_obj'].integers(low = 0, high = num_group_features, size = np.min([n_features, num_group_features]))
        # group_features = np.array(list(group_features))[group_feature_indices]
        group_features = adata.uns['seed_obj'].choice(np.array(list(group_features)), np.min([n_features, num_group_features]), replace = False) 

        # Create data arrays containing only features within this group
        X_train = adata[adata.uns['train_indices'],:][:, group_features].X
        X_test = adata[adata.uns['test_indices'],:][:, group_features].X


        X_train, X_test = process_data(X_train = X_train, X_test = X_test, data_type = adata.uns['data_type'], return_dense = True)

        #Extract pre-calculated sigma used for approximating kernel
        adjusted_sigma = adata.uns['sigma'][m]

        #Calculates approximate kernel according to chosen kernel function- may add more functions in the future
        #Distribution data comes from Fourier Transform of original kernel function
        if adata.uns['kernel_type'].lower() == 'gaussian':

            gamma = 1/(2*adjusted_sigma**2)
            sigma_p = 0.5*np.sqrt(2*gamma)

            W = adata.uns['seed_obj'].normal(0, sigma_p, X_train.shape[1]*D).reshape((X_train.shape[1]),D)

        elif adata.uns['kernel_type'].lower() == 'laplacian':

            gamma = 1/(2*adjusted_sigma)

            W = gamma * adata.uns['seed_obj'].standard_cauchy(X_train.shape[1]*D).reshape((X_train.shape[1],D))

        elif adata.uns['kernel_type'].lower() == 'cauchy':

            gamma = 1/(2*adjusted_sigma**2)
            b = 0.5*np.sqrt(gamma)

            W = adata.uns['seed_obj'].laplace(0, b, X_train.shape[1]*D).reshape((X_train.shape[1],D))


        train_projection = np.matmul(X_train, W)
        test_projection = np.matmul(X_test, W)
        

        #Store group Z in whole-Z object.  Preserves order to be able to extract meaningful groups
        Z_train[0:, np.arange( m * 2 * D , (m + 1) * 2 * D)] = np.sqrt(1/D)*np.hstack((np.cos(train_projection), np.sin(train_projection)))
        Z_test[0:, np.arange( m * 2 * D , (m + 1) * 2 * D)] = np.sqrt(1/D)*np.hstack((np.cos(test_projection), np.sin(test_projection)))

    adata.uns['Z_train'] = Z_train
    adata.uns['Z_test'] = Z_test


    return adata