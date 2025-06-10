import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD, PCA


def sparse_var(X, axis=None):
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


def process_data(X_train, X_test=None, scale_data=True, 
                  return_dense=True):
    '''
    Function to preprocess data matrix according to type of data 
    (counts- e.g. rna, or binary- atac). Will process test data 
    according to parameters calculated from test data
    
    Parameters
    ----------
    X_train : np.ndarray | scipy.sparse.matrix
        > A scipy sparse or numpy array of cells x features in the 
        training data.

    X_test : np.ndarray | scipy.sparse.matrix
        > A scipy sparse or numpy array of cells x features in the 
        testing data.

    scale_data : bool
        > If `True`, data will be logarithmized then z-score 
        transformed.

    return_dense: bool
        > If `True`, a np.ndarray will be returned as opposed to a 
        scipy.sparse object.
    
    Returns
    -------
    X_train, X_test : Numpy arrays with the process train/test data 
    respectively. If X_test is `None`, only X_train is returned.
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
    var = sparse_var(X_train, axis = 0)
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
    

def svd_transformation(X_train, X_test=None):
    '''
    Returns matrices with SVD reduction. If `X_test is None`, only 
    X_train is returned.

    Parameters
    ----------
    X_train : np.ndarray
        > A 2D array of cells x features filtered to desired features 
        for training data.

    X_test : np.ndarray
        > A 2D array of cells x features filtered to desired features 
        for testing data.
    
    Returns
    -------
    X_train, X_test : np.ndarray
        > Transformed matrices. Only X_train is returned if 
        `X_test is None`.
    '''
    n_components = np.min([50, X_train.shape[1]])
    SVD_func = TruncatedSVD(n_components = n_components, random_state = 1)
    
    # Remove first component as it corresponds with sequencing depth
    X_train = SVD_func.fit_transform(scipy.sparse.csr_array(X_train))[:, 1:]

    if X_test is not None:
        X_test = SVD_func.transform(scipy.sparse.csr_array(X_test))[:, 1:]
    
    return X_train, X_test


def pca_transformation(X_train, X_test=None):
    '''
    Returns matrices with PCA reduction. If `X_test is None`, only 
    X_train is returned.

    Parameters
    ----------
    X_train : np.ndarray
        > A 2D array of cells x features filtered to desired features 
        for training data.

    X_test : np.ndarray
        > A 2D array of cells x features filtered to desired features 
        for testing data.
    
    Returns
    -------
    X_train, X_test : np.ndarray
        > Transformed matrices. Only X_train is returned if 
        `X_test is None`.
    '''
    n_components = np.min([50, X_train.shape[1]])
    PCA_func = PCA(n_components = n_components, random_state = 1)

    X_train = PCA_func.fit_transform(np.asarray(X_train))

    if X_test is not None:
        X_test = PCA_func.transform(np.asarray(X_test))
    
    return X_train, X_test


def no_transformation(X_train, X_test=None):
    '''
    Dummy function used to return mat inputs.
    '''
    return X_train, X_test


def get_reduction(reduction: str):
    '''
    Function used to identify reduction type and return function to 
    apply to data matrices.
    '''
    match reduction:
        case 'pca':
            red_func = pca_transformation
        case 'svd':
            red_func = svd_transformation
        case 'None':
            red_func = no_transformation

    return red_func


def get_group_mat(adata, n_features, group_features, n_group_features, 
                  n_samples=None) -> np.ndarray:
    '''
    Filters to only features in group. Will sample features if 
    `n_features < n_group_features`.

    Parameters
    ----------
    adata : anndata.AnnData
        > anndata object with `'seed_obj'`, `'train_indices'`, and 
        `'test_indices'` in `.uns`.

    n_features : int
        > Maximum number of features to keep in matrix. Only 
        impacts mat if `n_features < n_group_features`.
    
    group_features : list | tuple | np.ndarray
        > Feature names in group to filter matrices to.

    n_group_features : int
        > Number of features in group.

    n_samples : int
        > Number of samples to filter X_train to.

    Returns
    -------
    X_train, X_test : np.ndarray
        > Filtered matrices. If `n_samples` is provided, only X_train 
        is returned. If `adata.uns['reduction']` is `'pca'` or `'svd'` 
        the matrices are transformed before being returned.
    '''
    # Getting reduction function
    reduction_func = get_reduction(adata.uns['reduction'])

    # Sample up to n_features features- important for scalability if 
    # using large groupings
    # Will use all features if the grouping contains fewer than n_features
    number_features = np.min([n_features, n_group_features])
    group_array = np.array(list(group_features))
    group_features = adata.uns['seed_obj'].choice(group_array, 
                                                  number_features, 
                                                  replace = False) 

    # Create data arrays containing only features within this group
    X_train = adata[adata.uns['train_indices'],:][:, group_features].X

    if n_samples is None:
        X_test = adata[adata.uns['test_indices'],:][:, group_features].X
        X_train, X_test = reduction_func(X_train, X_test)
        return X_train, X_test

    else:
        # Sample cells for scalability
        sample_idx = np.arange(X_train.shape[0])
        n_samples = np.min((n_samples, X_train.shape[0]))
        distance_indices = adata.uns['seed_obj'].choice(sample_idx, n_samples, 
                                                        replace = False)
        
        X_train = X_train[distance_indices]

        X_train, _ = reduction_func(X_train)

        return X_train