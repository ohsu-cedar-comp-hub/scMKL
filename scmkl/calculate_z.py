import numpy as np
import scipy
import anndata as ad

from scmkl.tfidf_normalize import tfidf_train_test
from scmkl.estimate_sigma import est_group_sigma, get_batches
from scmkl.data_processing import process_data, get_group_mat, sample_cells
from scmkl.projections import gaussian_trans, laplacian_trans, cauchy_trans


def get_z_indices(m, D):
    """
    Takes the number associated with the group as `m` and returns the 
    indices for cos and sin functions to be applied.

    Parameters
    ----------
    m : int
        The chronological number of the group being processed.

    D : int
        The number of dimensions per group.

    Returns
    -------
    cos_idx, sin_idx : np.ndarray, np.ndarray
        The indices for cos and sin projections in overall Z matrix.
    """
    x_idx = np.arange(m*2*D ,(m + 1)*2*D)
    cos_idx = x_idx[:len(x_idx)//2]
    sin_idx = x_idx[len(x_idx)//2:]

    return cos_idx, sin_idx


def calc_groupz(X_train, X_test, adata, D, sigma, proj_func):
    """
    Calculates the Z matrix for grouping.

    Parameters
    ----------
    X_train : np.ndarray
        The filtered data matrix to calculate train Z mat for.
    
    X_test : np.ndarray
        The filtered data matrix to calculate test Z mat for.

    adata : anndata.AnnData 
        AnnData object containing `seed_obj` in `.uns` attribute.

    D : int
        Number of dimensions per grouping.

    sigma : float
        Kernel width for grouping.

    proj_func : function
        The projection direction function to be applied to data.

    Returns
    -------
    train_projections, test_projections : np.ndarray, np.ndarray
        Training and testing Z matrices for group.
    """  
    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray().astype(np.float16)
        X_test = X_test.toarray().astype(np.float16)

    W = proj_func(X_train, sigma, adata.uns['seed_obj'], D)
    
    train_projection = np.matmul(X_train, W)
    test_projection = np.matmul(X_test, W)

    return train_projection, test_projection


def calculate_z(adata, n_features=5000, batches=10, 
                batch_size=100) -> ad.AnnData:
    """
    Function to calculate Z matrices for all groups in both training 
    and testing data.

    Parameters
    ----------
    adata : ad.AnnData
        created by `scmkl.create_adata()` with `adata.uns.keys()`: 
        `'train_indices'`, and `'test_indices'`. 

    n_features : int
        Number of random feature to use when calculating Z; used for 
        scalability.

    batches : int
        The number of batches to use for the distance calculation.
        This will average the result of `batches` distance calculations
        of `batch_size` randomly sampled cells. More batches will converge
        to population distance values at the cost of scalability.

    batch_size : int
        The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability.
        If `batches*batch_size > num_training_cells`,
        `batch_size` will be reduced to 
        `int(num_training_cells / batches)`.

    Returns
    -------
    adata : ad.AnnData
        `adata` with Z matrices accessible with `adata.uns['Z_train']` 
        and `adata.uns['Z_test']`.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata = scmkl.calculate_z(adata)
    >>> adata.uns.keys()
    dict_keys(['Z_train', 'Z_test', 'sigmas', 'train_indices', 
    'test_indices'])
    """
    # Number of groupings taking from group_dict
    n_pathway = len(adata.uns['group_dict'].keys())
    D = adata.uns['D']

    sq_i_d = np.sqrt(1/D)

    # Capturing training and testing sizes
    train_len = len(adata.uns['train_indices'])
    test_len = len(adata.uns['test_indices'])

    if batch_size * batches > len(adata.uns['train_indices']):
        old_batch_size = batch_size
        batch_size = int(len(adata.uns['train_indices'])/batches)
        print("Specified batch size required too many cells for "
                "independent batches. Reduced batch size from "
                f"{old_batch_size} to {batch_size}")

    if 'sigma' not in adata.uns.keys():
        n_samples = np.min((2000, adata.uns['train_indices'].shape[0]))
        sample_range = np.arange(n_samples)
        batch_idx = get_batches(sample_range, adata.uns['seed_obj'], 
                                batches=batches, batch_size=batch_size)
        sigma_indices = sample_cells(adata.uns['train_indices'], n_samples, adata.uns['seed_obj'])

    # Create Arrays to store concatenated group Zs
    # Each group of features will have a corresponding entry in each array
    n_cols = 2*adata.uns['D']*n_pathway
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
                                        n_group_features, process_test=True)
        
        if adata.uns['tfidf']:
            X_train, X_test = tfidf_train_test(X_train, X_test)

        # Data filtering, and transformation according to given data_type
        # Will remove low variance (< 1e5) features regardless of data_type
        # If scale_data will log scale and z-score the data
        X_train, X_test = process_data(X_train=X_train, X_test=X_test, 
                                       scale_data=adata.uns['scale_data'], 
                                       return_dense=True)    

        # Getting sigma
        if 'sigma' in adata.uns.keys():
            sigma = adata.uns['sigma'][m]
        else:
            sigma = est_group_sigma(adata, X_train, n_group_features, 
                                    n_features, batch_idx=batch_idx)
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

    adata.uns['Z_train'] = Z_train*sq_i_d
    adata.uns['Z_test'] = Z_test*sq_i_d

    if 'sigma' not in adata.uns.keys():
        adata.uns['sigma'] = np.array(sigma_list)

    return adata