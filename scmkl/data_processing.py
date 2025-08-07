import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD, PCA
import anndata as ad


def sparse_var(X: scipy.sparse._csc.csc_matrix | np.ndarray, axis: int | None=None):
    """
    Function to calculate variance on a scipy sparse matrix.
    
    Parameters
    ----------
    X : scipy.sparse._csc.csc_matrix | np.ndarray
        A scipy sparse or numpy array
        
    axis : int | None
        Determines which axis variance is calculated on. Same usage 
        as Numpy.
    
    Returns
    -------
    var : np.ndarray | float
        Variance values calculated over the given axis.
    """
    # E[X^2] - E[X]^2
    if scipy.sparse.issparse(X):
        exp_mean = np.asarray(X.power(2).mean(axis = axis)).flatten()
        sq_mean = np.asarray(np.square(X.mean(axis = axis))).flatten()
        var = np.array(exp_mean - sq_mean)
    else:
        var = np.asarray(np.var(X, axis = axis)).flatten()

    return var.ravel()


def process_data(X_train: np.ndarray | scipy.sparse._csc.csc_matrix,
                 X_test: np.ndarray | scipy.sparse._csc.csc_matrix | None=None,
                 scale_data: bool=True, 
                 return_dense: bool=True):
    """
    Function to preprocess data matrix according to type of data 
    (e.g. counts/rna, or binary/atac). Will process test data 
    according to parameters calculated from test data.
    
    Parameters
    ----------
    X_train : np.ndarray | scipy.sparse._csc.csc_matrix
        A scipy sparse or numpy array of cells x features in the 
        training data.

    X_test : np.ndarray | scipy.sparse._csc.csc_matrix
        A scipy sparse or numpy array of cells x features in the 
        testing data.

    scale_data : bool
        If `True`, data will be logarithmized then z-score 
        transformed.

    return_dense: bool
        If `True`, a np.ndarray will be returned as opposed to a 
        scipy.sparse object.
    
    Returns
    -------
    X_train, X_test : np.ndarray, np.ndarray
        Numpy arrays with the process train/test data 
        respectively. If X_test is `None`, only X_train is returned.
    """
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
    

def svd_transformation(X_train: scipy.sparse._csc.csc_matrix | np.ndarray,
                       X_test: scipy.sparse._csc.csc_matrix | 
                       np.ndarray | None=None):
    """
    Returns matrices with SVD reduction. If `X_test is None`, only 
    X_train is returned.

    Parameters
    ----------
    X_train : np.ndarray
        A 2D array of cells x features filtered to desired features 
        for training data.

    X_test : np.ndarray | None
        A 2D array of cells x features filtered to desired features 
        for testing data.
    
    Returns
    -------
    X_train, X_test : np.ndarray, np.ndarray
        Transformed matrices. Only X_train is returned if 
        `X_test is None`.
    """
    n_components = np.min([50, X_train.shape[1]])
    SVD_func = TruncatedSVD(n_components = n_components, random_state = 1)
    
    # Remove first component as it corresponds with sequencing depth
    # We convert to a csr_array because the SVD function is faster on this
    # matrix type
    X_train = SVD_func.fit_transform(scipy.sparse.csr_array(X_train))[:, 1:]

    if X_test is not None:
        X_test = SVD_func.transform(scipy.sparse.csr_array(X_test))[:, 1:]
    
    return X_train, X_test


def sample_cells(train_indices: np.ndarray,
                 sample_size: int,
                 seed_obj: np.random._generator.Generator):
    """
    Samples cells indices from training indices for calculations.

    Parameters
    ----------
    train_indices : np.ndarray
        An array of indices to sample from.

    sample_size : int
        Number of samples to take from `train_indices`. Must be 
        smaller than length of `train_indices`.

    Returns
    -------
    indices : np.ndarray
        The sampled indices from `train_indices`.
    """
    n_samples = np.min((train_indices.shape[0], sample_size))
    indices = seed_obj.choice(train_indices, n_samples, replace = False)

    return indices


def pca_transformation(X_train: scipy.sparse._csc.csc_matrix | np.ndarray,
                       X_test: scipy.sparse._csc.csc_matrix | np.ndarray | None=None):
    """
    Returns matrices with PCA reduction. If `X_test is None`, only 
    X_train is returned.

    Parameters
    ----------
    X_train : scipy.sparse._csc.csc_matrix | np.ndarray
        A 2D array of cells x features filtered to desired features 
        for training data.

    X_test : scipy.sparse._csc.csc_matrix | np.ndarray | None
        A 2D array of cells x features filtered to desired features 
        for testing data.
    
    Returns
    -------
    X_train, X_test : np.ndarray, np.ndarray
        Transformed matrices. Only X_train is returned if 
        `X_test is None`.
    """
    n_components = np.min([50, X_train.shape[1]])
    PCA_func = PCA(n_components = n_components, random_state = 1)

    X_train = PCA_func.fit_transform(np.asarray(X_train))

    if X_test is not None:
        X_test = PCA_func.transform(np.asarray(X_test))
    
    return X_train, X_test


def _no_transformation(X_train: scipy.sparse._csc.csc_matrix | np.ndarray,
                      X_test: scipy.sparse._csc.csc_matrix | np.ndarray | None=None):
    """
    Dummy function used to return mat inputs.
    """
    return X_train, X_test


def get_reduction(reduction: str):
    """
    Function used to identify reduction type and return function to 
    apply to data matrices.

    Parameters
    ----------
    reduction : str
        The reduction for data transformation. Options are `['pca', 
        'svd', 'None']`.

    Returns
    -------
    red_func : function
        The function to reduce the data.
    """
    match reduction:
        case 'pca':
            red_func = pca_transformation
        case 'svd':
            red_func = svd_transformation
        case 'None':
            red_func = _no_transformation

    return red_func


def get_group_mat(adata: ad.AnnData, n_features: int,
                  group_features: np.ndarray,
                  n_group_features: int, 
                  process_test: bool=False) -> np.ndarray:
    """
    Filters to only features in group. Will sample features if 
    `n_features < n_group_features`.

    Parameters
    ----------
    adata : anndata.AnnData
        anndata object with `'seed_obj'`, `'train_indices'`, and 
        `'test_indices'` in `.uns`.

    n_features : int
        Maximum number of features to keep in matrix. Only 
        impacts mat if `n_features < n_group_features`.
    
    group_features : list | tuple | np.ndarray
        Feature names in group to filter matrices to.

    n_group_features : int
        Number of features in group.

    n_samples : int
        Number of samples to filter X_train to.

    Returns
    -------
    X_train, X_test : np.ndarray, np.ndarray
        Filtered matrices. If `n_samples` is provided, only `X_train` 
        is returned. If `adata.uns['reduction']` is `'pca'` or 
        `'svd'` the matrices are transformed before being returned.
    """
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
    if process_test:
        X_train = adata[adata.uns['train_indices'],:][:, group_features].X
        X_test = adata[adata.uns['test_indices'],:][:, group_features].X
        X_train, X_test = reduction_func(X_train, X_test)
        return X_train, X_test

    else:
        X_train = adata[:, group_features].X
        return X_train