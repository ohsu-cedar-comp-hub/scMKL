import numpy as np
import scipy
import anndata as ad


def _sparse_var(X, axis = None):

    '''
    Function to calculate variance on a scipy sparse matrix.
    
    Input:
        X- A scipy sparse or numpy array
        axis- Determines which axis variance is calculated on. Same usage as Numpy
            axis = 0 => column variances
            axis = 1 => row variances
            axis = None => total variance (calculated on all data)
    Output:
        var- Variance values calculated over the given axis
    '''

    # E[X^2] - E[X]^2
    if scipy.sparse.issparse(X):
        var = np.array((X.power(2).mean(axis = axis)) - np.square(X.mean(axis = axis)))
    else:
        var = np.var(X, axis = axis)
    return var.ravel()


def _process_data(X_train, X_test = None, scale_data = True, return_dense = True):
    '''
    Function to preprocess data matrix according to type of data (counts- e.g. rna, or binary- atac)
    Will process test data according to parameters calculated from test data

    Input:
        X_train- A scipy sparse or numpy array
        X_train- A scipy sparse or numpy array
        data_type- 'counts' or 'binary'.  Determines what preprocessing is applied to the data. 
            Log transforms and standard scales counts data
            TFIDF filters ATAC data to remove uninformative columns
    Output:
        X_train, X_test- Numpy arrays with the process train/test data respectively.
    '''


    if X_test is None:
            X_test = X_train[:1,:] # Creates dummy matrix to for the sake of calculation without increasing computational time
            orig_test = None
    else:
        orig_test = 'given'

    # Remove features that have no variance in the training data (will be uniformative)
    var = _sparse_var(X_train, axis = 0)
    variable_features = np.where(var > 1e-5)[0]

    X_train = X_train[:,variable_features]
    X_test = X_test[:, variable_features]

    #Data processing according to data type
    if scale_data:

        if scipy.sparse.issparse(X_train):
            X_train = X_train.log1p()
            X_test = X_test.log1p()
        else:
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)
            
        #Center and scale count data
        train_means = np.mean(X_train, 0)
        train_sds = np.sqrt(var[variable_features])

        # Perform transformation on test data according to parameters of the training data
        X_train = (X_train - train_means) / train_sds
        X_test = (X_test - train_means) / train_sds

    if return_dense and scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    if orig_test == None:
        return X_train
    else:
        return X_train, X_test

def calculate_z(adata, n_features = 5000) -> ad.AnnData:
    '''
    Function to calculate Z matrix.

    Parameters
    ----------
    **adata** : *AnnData*
        > created by `create_adata()` with `adata.uns.keys()` `'sigma'`, 
        `'train_indices'`, and `'test_indices'`. `'sigma'` key can be 
        added by running `estimate_sigma()` on adata. 

    **n_features** : *int* 
        > Number of random feature to use when calculating Z- used for 
        scalability.

    Returns
    -------
    **adata** : *AnnData*
        > adata with Z matrices accessible with `adata.uns['Z_train']` 
        and `adata.uns['Z_test']`.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata = scmkl.calculate_z(adata)
    >>> adata.uns.keys()
    dict_keys(['Z_train', 'Z_test', 'sigmas', 'train_indices', 'test_indices])
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

        # Sample up to n_features features- important for scalability if using large groupings
        # Will use all features if the grouping contains fewer than n_features
        group_features = adata.uns['seed_obj'].choice(np.array(list(group_features)), np.min([n_features, num_group_features]), replace = False) 

        # Create data arrays containing only features within this group
        X_train = adata[adata.uns['train_indices'],:][:, group_features].X
        X_test = adata[adata.uns['test_indices'],:][:, group_features].X

        # Perform data filtering, and transformation according to given data_type
        # Will remove low variance (< 1e5) features regardless of data_type
        # If given data_type is 'counts' (like RNA) will log scale and z-score the data
        X_train, X_test = _process_data(X_train = X_train, X_test = X_test, scale_data = adata.uns['scale_data'], return_dense = True)

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