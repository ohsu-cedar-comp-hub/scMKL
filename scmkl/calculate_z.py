import numpy as np
import scipy
import anndata as ad

from scmkl.tfidf_normalize import _tfidf_train_test
from scmkl.estimate_sigma import est_group_sigma
from scmkl.data_processing import process_data, get_group_mat
from scmkl.projections import gaussian_trans, laplacian_trans, cauchy_trans


def get_z_indices(m, D):
    '''
    Takes the number associated with the group as `m` and returns the 
    indices for cos and sin functions to be applied.

    Parameters
    ----------
    m : int
        > The chronological number of the group being processed.

    D : int
        > The number of dimensions per group.

    Returns
    -------
    cos_idx, sin_idx : np.ndarray
        > The indices for cos and sin projections in overall Z matrix.
    '''
    x_idx = np.arange(m * 2 * D ,(m + 1) * 2 * D)
    cos_idx = x_idx[:len(x_idx)//2]
    sin_idx = x_idx[len(x_idx)//2:]

    return cos_idx, sin_idx


def calc_groupz(X_train, X_test, adata, D, sigma, proj_func):
    '''
    Calculates the Z matrix for grouping.

    Parameters
    ----------
    X_train : np.ndarray
        > The filtered data matrix to calculate train Z mat for.
    
    X_train : np.ndarray
        > The filtered data matrix to calculate train Z mat for.

    adata : anndata.AnnData 
        > anndata object containing `seed_obj` in `.uns` attribute.

    D : int
        > Number of dimensions per grouping.

    sigma : float
        > Kernel width for grouping.

    proj_func : function
        > The projection direction function to be applied to data.

    Returns
    -------
    train_projections, test_projections : np.ndarray
        > Training and testing Z matrices for group.
    '''   
    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray().astype(np.float16)
        X_test = X_test.toarray().astype(np.float16)

    w = proj_func(X_train, sigma, adata.uns['seed_obj'], D)
    
    train_projection = np.matmul(X_train, w)
    test_projection = np.matmul(X_test, w)

    return train_projection, test_projection


def calculate_z(adata, n_features = 5000, batches = 10, 
                batch_size = 100) -> ad.AnnData:
    '''
    Function to calculate Z matrices for all groups in both training 
    and testing data.

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
    # Number of groupings taking from group_dict
    n_pathway = len(adata.uns['group_dict'].keys())
    D = adata.uns['D']

    sq_i_d = np.sqrt(1/D)

    # Capturing training and testing indices
    train_len = len(adata.uns['train_indices'])
    test_len = len(adata.uns['test_indices'])

    # Create Arrays to store concatenated group Zs
    # Each group of features will have a corresponding entry in each array
    n_cols = 2 * adata.uns['D'] * n_pathway
    Z_train = np.zeros((train_len, n_cols))
    Z_test = np.zeros((test_len, n_cols))


    # Setting kernel function 
    match adata.uns['kernel_type'].lower():
        case 'gaussian':
            proj_func = gaussian_trans
        case 'laplacian':
            proj_func = laplacian_trans
        case 'cauchy':
            proj_func = cauchy_trans


    # Loop over each of the groups and creating Z for each
    sigma_list = list()
    for m, group_features in enumerate(adata.uns['group_dict'].values()):

        n_group_features = len(group_features)

        X_train, X_test = get_group_mat(adata, n_features, group_features, 
                                        n_group_features)
        
        if adata.uns['tfidf']:
            X_train, X_test = _tfidf_train_test(X_train, X_test)

        # Data filtering, and transformation according to given data_type
        # Will remove low variance (< 1e5) features regardless of data_type
        # If scale_data will log scale and z-score the data
        X_train, X_test = process_data(X_train=X_train, X_test=X_test, 
                                       scale_data=adata.uns['scale_data'], 
                                       return_dense = True)    

        # Getting sigma
        if 'sigma' in adata.uns.keys():
            sigma = adata.uns['sigma'][m]
        else:
            sigma = est_group_sigma(adata, X_train, n_group_features, 
                                    n_features, batches, batch_size)
            sigma_list.append(sigma)
            
        assert sigma > 0, "Sigma must be more than 0"
        train_projection, test_projection = calc_groupz(X_train, X_test, 
                                                        adata, D, sigma, 
                                                        proj_func)

        # Store group Z in whole-Z object
        # Preserves order to be able to extract meaningful groups
        cos_idx, sin_idx = get_z_indices(m, D)

        Z_train[0:, cos_idx] = np.cos(train_projection)
        Z_train[0:, sin_idx] = np.sin(train_projection)

        Z_test[0:, cos_idx] = np.cos(test_projection)
        Z_test[0:, sin_idx] = np.sin(test_projection)

    adata.uns['Z_train'] = Z_train * sq_i_d
    adata.uns['Z_test'] = Z_test * sq_i_d

    if 'sigma' not in adata.uns.keys():
        adata.uns['sigma'] = np.array(sigma_list)

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