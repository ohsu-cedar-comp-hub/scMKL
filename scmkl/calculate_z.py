import numpy as np
import scipy
import anndata as ad
from sklearn.decomposition import TruncatedSVD, PCA
from scmkl.tfidf_normalize import _tfidf_train_test

def _sparse_var(X, axis = None):
    '''
    Function to calculate variance on a scipy sparse matrix.
    
    Parameters
    ----------
    X : A scipy sparse or numpy array
    axis : Determines which axis variance is calculated on. Same usage 
    as Numpy.
        axis = 0 => column variances
        axis = 1 => row variances
        axis = None => total variance (calculated on all data)
    
    Returns
    -------
    var : Variance values calculated over the given axis
    '''

    # E[X^2] - E[X]^2
    if scipy.sparse.issparse(X):
        exp_mean = (X.power(2).mean(axis = axis))
        sq_mean = np.square(X.mean(axis = axis))
        var = np.array(exp_mean - sq_mean)
    else:
        var = np.var(X, axis = axis)

    return var.ravel()


def _process_data(X_train, X_test = None, scale_data = True, 
                  return_dense = True):
    '''
    Function to preprocess data matrix according to type of data 
    (counts- e.g. rna, or binary- atac). Will process test data 
    according to parameters calculated from test data
    
    Parameters
    ----------
    X_train : A scipy sparse or numpy array
    X_train : A scipy sparse or numpy array
    data_type : 'counts' or 'binary'.  Determines what preprocessing is 
                applied to the data. Log transforms and standard scales 
                counts data TFIDF filters ATAC data to remove 
                uninformative columns
    
    Returns
    -------
    X_train, X_test : Numpy arrays with the process train/test data 
    respectively.
    '''
    if X_test is None:
            # Creates dummy matrix to for the sake of calculation without 
            # increasing computational time
            X_test = X_train[:1,:] 
            orig_test = None
    else:
        orig_test = 'given'

    # Remove features that have no variance in the training data 
    # (will be uniformative)
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

        # Perform transformation on test data according to parameters 
        # of the training data
        X_train = (X_train - train_means) / train_sds
        X_test = (X_test - train_means) / train_sds


    if return_dense and scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()


    if orig_test is None:
        return X_train
    else:
        return X_train, X_test


def calculate_z(adata, n_features = 5000) -> ad.AnnData:
    '''
    Function to calculate Z matrix.

    Parameters
    ----------
    **adata** : *AnnData*
        > created by `create_adata()` with `adata.uns.keys()` 
        `'sigma'`, `'train_indices'`, and `'test_indices'`. 
        `'sigma'` key can be added by running `estimate_sigma()` on 
        adata. 

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
    dict_keys(['Z_train', 'Z_test', 'sigmas', 'train_indices', 
    'test_indices'])
    '''
    assert np.all(adata.uns['sigma'] > 0), 'Sigma must be positive'

    # Number of groupings taking from group_dict
    n_pathway = len(adata.uns['group_dict'].keys())
    D = adata.uns['D']

    # Capturing training and testing indices
    train_idx = np.array(adata.uns['train_indices'], dtype = np.int_)
    test_idx = np.array(adata.uns['test_indices'], dtype = np.int_)

    # Create Arrays to store concatenated group Z
    # Each group of features will have a corresponding entry in each array
    n_cols = 2 * adata.uns['D'] * n_pathway

    Z_train = np.zeros((train_idx.shape[0], n_cols))
    Z_test = np.zeros((test_idx.shape[0], n_cols))


    # Loop over each of the groups and creating Z for each
    for m, group_features in enumerate(adata.uns['group_dict'].values()):
        
        #Extract features from mth group
        num_group_features = len(group_features)

        # Sample up to n_features features- important for scalability if 
        # using large groupings
        # Will use all features if the grouping contains fewer than n_features
        number_features = np.min([n_features, num_group_features])
        group_array = np.array(list(group_features))
        group_features = adata.uns['seed_obj'].choice(group_array, 
                                                      number_features, 
                                                      replace = False) 

        # Create data arrays containing only features within this group
        X_train = adata[adata.uns['train_indices'],:][:, group_features].X
        X_test = adata[adata.uns['test_indices'],:][:, group_features].X

        if adata.uns['tfidf']:
            X_train, X_test = _tfidf_train_test(X_train, X_test)

        # Data filtering, and transformation according to given data_type
        # Will remove low variance (< 1e5) features regardless of data_type
        # If given data_type is 'counts' will log scale and z-score the data
        X_train, X_test = _process_data(X_train = X_train, X_test = X_test, 
                                        scale_data = adata.uns['scale_data'], 
                                        return_dense = True)          

        if adata.uns['reduction'].lower() == 'svd':

            SVD_func = TruncatedSVD(n_components = np.min([50, X_train.shape[1]]), random_state = 1)
            
            # Remove first component as it corresponds with sequencing depth
            X_train = SVD_func.fit_transform(scipy.sparse.csr_array(X_train))[:, 1:]
            X_test = SVD_func.transform(scipy.sparse.csr_array(X_test))[:, 1:]

        elif adata.uns['reduction'].lower() == 'pca':
            PCA_func = PCA(n_components = np.min([50, X_train.shape[1]]), random_state = 1)

            X_train = PCA_func.fit_transform(np.asarray(X_train))
            X_test = PCA_func.transform(np.asarray(X_test))

        elif adata.uns['reduction'] == 'linear':

            X_train = X_train @ adata.uns['seed_obj'].choice([0,1], p = [0.02, 0.98], size = X_train.shape[1] * 50).reshape((X_train.shape[1], 50))
            X_test = X_test @ adata.uns['seed_obj'].choice([0,1], p = [0.02, 0.98], size = X_test.shape[1] * 50).reshape((X_train.shape[1], 50))

        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray().astype(np.float16)
            X_test = X_test.toarray().astype(np.float16)

        # Extract pre-calculated sigma used for approximating kernel
        adjusted_sigma = adata.uns['sigma'][m]

        # Calculates approximate kernel according to chosen kernel function
        # Distribution data comes from Fourier Transform of kernel function
        if adata.uns['kernel_type'].lower() == 'gaussian':

            gamma = 1/(2*adjusted_sigma**2)
            sigma_p = 0.5*np.sqrt(2*gamma)

            W = adata.uns['seed_obj'].normal(0, sigma_p, X_train.shape[1] * D)
            W = W.reshape((X_train.shape[1]), D)

        elif adata.uns['kernel_type'].lower() == 'laplacian':

            gamma = 1 / (2 * adjusted_sigma)

            W = adata.uns['seed_obj'].standard_cauchy(X_train.shape[1] * D)
            W = gamma * W.reshape((X_train.shape[1], D))

        elif adata.uns['kernel_type'].lower() == 'cauchy':

            gamma = 1 / (2 * adjusted_sigma ** 2)
            b = 0.5 * np.sqrt(gamma)

            W = adata.uns['seed_obj'].laplace(0, b, X_train.shape[1] * D)
            W = W.reshape((X_train.shape[1], D))


        train_projection = np.matmul(X_train, W)
        test_projection = np.matmul(X_test, W)
        
        # Store group Z in whole-Z object. 
        # Preserves order to be able to extract meaningful groups
        x_idx = np.arange( m * 2 * D , (m + 1) * 2 * D)
        sq_i_d = np.sqrt(1/D)

        Z_train[0:, x_idx] = sq_i_d * np.hstack((np.cos(train_projection), 
                                                 np.sin(train_projection)))
        Z_test[0:, x_idx] = sq_i_d * np.hstack((np.cos(test_projection), 
                                                np.sin(test_projection)))

    adata.uns['Z_train'] = Z_train
    adata.uns['Z_test'] = Z_test

    return adata