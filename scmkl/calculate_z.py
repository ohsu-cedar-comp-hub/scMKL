import numpy as np
import scipy
import anndata as ad
from sklearn.decomposition import TruncatedSVD, PCA

from scmkl.tfidf_normalize import _tfidf_train_test
from scmkl.kernels import *


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

    # Data processing according to data type
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

    sq_i_d = np.sqrt(1/D)

    # Capturing training and testing indices
    train_idx = np.array(adata.uns['train_indices'], dtype = np.int_)
    test_idx = np.array(adata.uns['test_indices'], dtype = np.int_)

    # Create Arrays to store concatenated group Z
    # Each group of features will have a corresponding entry in each array
    n_cols = 2 * adata.uns['D'] * n_pathway

    Z_train = np.zeros((train_idx.shape[0], n_cols))
    Z_test = np.zeros((test_idx.shape[0], n_cols))


    # Setting kernel function 
    match adata.uns['kernel_type'].lower():
        case 'gaussian':
            kernel_func = gaussian_trans
        case 'laplacian':
            kernel_func = laplacian_trans
        case 'cauchy':
            kernel_func = cauchy_trans


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
        # If scale_data will log scale and z-score the data
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


        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray().astype(np.float16)
            X_test = X_test.toarray().astype(np.float16)

        # Extract pre-calculated sigma used for approximating kernel
        adjusted_sigma = adata.uns['sigma'][m]

        w = kernel_func(X_train, adjusted_sigma, adata.uns['seed_obj'], D)


        train_projection = np.matmul(X_train, w)
        test_projection = np.matmul(X_test, w)
        
        # Store group Z in whole-Z object. 
        # Preserves order to be able to extract meaningful groups
        x_idx = np.arange(m * 2 * D ,(m + 1) * 2 * D)
        cos_idx = x_idx[:len(x_idx)//2]
        sin_idx = x_idx[len(x_idx)//2:]

        Z_train[0:, cos_idx] = np.cos(train_projection)
        Z_train[0:, sin_idx] = np.sin(train_projection)

        Z_test[0:, cos_idx] = np.cos(test_projection)
        Z_test[0:, sin_idx] = np.sin(test_projection)

    adata.uns['Z_train'] = Z_train * sq_i_d
    adata.uns['Z_test'] = Z_test * sq_i_d

    return adata


def transform_z(adata, new_sigmas)-> ad.AnnData:

    '''
    This functions takes an adata object with Z_train and Z_test already 
    calculated and transforms it as if calculated with a different distribution.

    This is primarily used during optimize_alpha to remove dependence on fold
    train data without need to recalculate Z_train and Z_test.

    i.e. (X @ W_old) * old_sigma / new_sigma == (X @ W_new) (the inverse relationship
    between the distribution and the sigma parameter is because the standard deviation
    of the distribution is proportional to 1/sigma).
    '''

    assert 'Z_train' in adata.uns.keys() and 'Z_test' in adata.uns.keys(), 'Z_train and Z_test must be present in adata'
    assert all(new_sigmas > 0) and all(adata.uns['sigma'] > 0), 'Sigma must be positive'
    assert len(new_sigmas) == len(adata.uns['sigma']), 'Length of new sigmas must be equal to length of old sigmas'

    Z_train = adata.uns['Z_train']
    Z_test = adata.uns['Z_test']


    for i, (sigma, new_sigma) in enumerate(zip(adata.uns['sigma'], new_sigmas)):
        if sigma == new_sigma:
            continue

        group_idx = np.arange(i * 2 * adata.uns['D'], (i + 1) * 2 * adata.uns['D'])
        cos_idx = group_idx[:len(group_idx)//2]
        sin_idx = group_idx[len(group_idx)//2:]

        # Undo the cos/sin and transform the recovered X @ W and transforms the distribution
        cos_train = Z_train[:, cos_idx]
        cos_test = Z_test[:, cos_idx]
        sin_train = Z_train[:, sin_idx]
        sin_test = Z_test[:, sin_idx]

        orig_train_projection = np.arctan2(sin_train, cos_train)
        orig_test_projection = np.arctan2(sin_test, cos_test)

        
        transformed_train_projection = orig_train_projection * (sigma / new_sigma)
        transformed_test_projection = orig_test_projection * (sigma / new_sigma)


        Z_train[:, cos_idx] = np.cos(transformed_train_projection)
        Z_test[:, cos_idx] = np.cos(transformed_test_projection)

        Z_train[:, sin_idx] = np.sin(transformed_train_projection)
        Z_test[:, sin_idx] = np.sin(transformed_test_projection)

    adata.uns['Z_train'] = Z_train
    adata.uns['Z_test'] = Z_test

    return adata