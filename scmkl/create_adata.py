import numpy as np
import anndata as ad


def _filter_features(X, feature_names, group_dict):
    '''
    Function to remove unused features from X matrix.  Any features not included in group_dict will be removed from the matrix.
    Also puts the features in the same relative order (of included features)
    Input:
            X- Data array. Can be Numpy array or Scipy Sparse Array
            feature_names- Numpy array of corresponding feature names
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
    Output:
            X- Data array containing data only for features in the group_dict
            feature_names- Numpy array of corresponding feature names from group_dict
    '''
    assert X.shape[1] == len(feature_names), 'Given features do not correspond with features in X'    

    group_features = set()
    feature_set = set(feature_names)

    # Store all objects in dictionary in set
    for group in group_dict.keys():
        group_features.update(set(group_dict[group]))

        # Finds intersection between group features and features in data
        # Converts to numpy array and sorts to preserve order of feature names
        group_dict[group] = np.sort(np.array(list(feature_set.intersection(set(group_dict[group])))))

    # Find location of desired features in whole feature set
    group_feature_indices = np.where(np.in1d(feature_names, np.array(list(group_features)), assume_unique = True))[0]

    # Subset only the desired features and their data
    X = X[:,group_feature_indices]
    feature_names = np.array(list(feature_names))[group_feature_indices]

    return X, feature_names, group_dict


def _train_test_split(y, train_indices = None, seed_obj = np.random.default_rng(100), train_ratio = 0.8):
    '''
    Function to calculate training and testing indices for given dataset. If train indices are given, it will calculate the test indices.
        If train_indices == None, then it calculates both indices, preserving the ratio of each label in y
    Input:
            y- Numpy array of cell labels. Can have any number of classes for this function.
            train_indices- Optional array of pre-determined training indices
            seed_obj- Numpy random state used for random processes. Can be specified for reproducubility or set by default.
            train_ratio- decimal value ratio of features in training:testing sets
    Output:
            train_indices- Array of indices of training cells
            test_indices- Array of indices of testing cells
    '''

    # If train indices aren't provided
    if train_indices is None:

        unique_labels = np.unique(y)
        train_indices = []

        for label in unique_labels:

            # Find index of each unique label
            label_indices = np.where(y == label)[0]

            # Sample these indices according to train ratio
            train_label_indices = seed_obj.choice(label_indices, int(len(label_indices) * train_ratio), replace = False)
            train_indices.extend(train_label_indices)
    else:
        assert len(train_indices) <= len(y), 'More train indices than there are samples'

    train_indices = np.array(train_indices)

    # Test indices are the indices not in the train_indices
    test_indices = np.setdiff1d(np.arange(len(y)), train_indices, assume_unique = True)

    return train_indices, test_indices


def create_adata(X, feature_names: np.ndarray, cell_labels: np.ndarray, group_dict: dict, scale_data: bool = True, split_data = None, D = 100, 
                 remove_features = True, distance_metric = 'euclidean', kernel_type = 'Gaussian', random_state = 1):
    
    '''
    Function to create an AnnData object to carry all relevant 
    information going forward.

    Parameters
    ----------
    **X** : *scipy.sparse.csc_matrix* | *np.ndarray* | 
            *pd.DataFrame*
        > A data matrix of cells by features (sparse array 
        recommended for large datasets).

    **feature_names** : *np.ndarray*
        > array of feature names corresponding with the features 
        in X.

    **cell_labels** : *np.ndarray*
        > A numpy array of cell phenotypes corresponding with 
        the cells in X.

    **group_dict** : *dict* 
        > Dictionary containing feature grouping information.
            - Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}

    **scale_data** : *bool*  
        > If `True`, data matrix is log transformed and standard 
        scaled. 
        
    **split_data** : *None* | *np.ndarray*
        > If *None*, data will be split stratified by cell labels. 
        Else, is an array of precalculated train/test split 
        corresponding to samples.

    **D** : *int* 
        > Number of Random Fourier Features used to calculate Z. 
        Should be a positive integer. Higher values of D will 
        increase classification accuracy at the cost of computation 
        time.
    
    **remove_features** : *bool* 
        > If `True`, will remove features from X and feature_names
        not in group_dict and remove features from groupings not in
        feature_names.

    **distance_metric** : *str* 
        > The pairwise distance metric used to estimate sigma. Must
        be one of the options used in scipy.spatial.distance.cdist

    **kernel_type** : *str*
        > The approximated kernel function used to calculate Zs.
        Must be one of `'Gaussian'`, `'Laplacian'`, or `'Cauchy'`.

    **random_state** : *int*
        > Integer random_state used to set the seed for 
        reproducibilty.

    Returns
    -------
    **adata** : *AnnData*
    > *AnnData* with the following attributes and keys:

    > `adata.X` : the data matrix
    
    > `adata.var_names` : the feature names corresponding to
    `adata.X`.

    > `adata.obs['labels']` : cell classes/phenotypes from 
    `cell_labels`.

    > `adata.uns['train_indices']` : Indices for training data. 

    > `adata.uns['test_indices']` : Indices for testing data.

    > `adata.uns['group_dict']` : Grouping information.

    > `adata.uns['seed_obj']` : Seed object with seed equal to
    100 * `random_state`.

    > `with adata.uns['D']` : Number of dimensions to scMKL with.

    > `adata.uns['data_type']` : *bool* for wether or not data
    is log transformed and scaled.

    > `adata.uns['distance_metric']` : Distance metric as given.
    
    > `adata.uns['kernel_type']` : Kernel function as given.

    Examples
    --------
    >>> data_mat = scipy.sparse.load_npz('MCF7_RNA_matrix.npz')
    >>> gene_names = np.load('MCF7_gene_names.pkl', allow_pickle = True)
    >>> group_dict = np.load('hallmark_genesets.pkl', allow_pickle = True)
    >>> 
    >>> adata = scmkl.create_adata(X = data_mat, 
    ...                            feature_names = gene_names, 
    ...                            group_dict = group_dict)
    >>> adata
    AnnData object with n_obs × n_vars = 1000 × 4341
    obs: 'labels'
    uns: 'group_dict', 'seed_obj', 'scale_data', 'D', 'kernel_type', 
    'distance_metric', 'train_indices', 'test_indices'
    '''

    assert X.shape[0] == len(cell_labels), 'Different number of cells than labels'
    assert X.shape[1] == len(feature_names), 'Different number of features in X than feature names'
    assert len(np.unique(cell_labels)) == 2, 'cell_labels must contain 2 classes'
    assert isinstance(D, int) and D > 0, 'D must be a positive integer'
    assert kernel_type.lower() in ['gaussian', 'laplacian', 'cauchy'], 'Given kernel type not implemented. Gaussian, Laplacian, and Cauchy are the acceptable types.'

    if remove_features:
        X, feature_names, group_dict = _filter_features(X, feature_names, group_dict)

    # Create adata object and add column names
    adata = ad.AnnData(X)
    adata.var_names = feature_names

    # Add metadata to adata object
    adata.obs['labels'] = cell_labels
    adata.uns['group_dict'] = group_dict
    adata.uns['seed_obj'] = np.random.default_rng(100 * random_state)
    adata.uns['scale_data'] = scale_data
    adata.uns['D'] = D
    adata.uns['kernel_type'] = kernel_type
    adata.uns['distance_metric'] = distance_metric

    if split_data == None:
        train_indices, test_indices = _train_test_split(cell_labels, seed_obj = adata.uns['seed_obj'])
    else:
        train_indices = np.where(split_data == 'train')[0]
        test_indices = np.where(split_data == 'test')[0]

    adata.uns['train_indices'] = train_indices
    adata.uns['test_indices'] = test_indices

    if not scale_data:
        print('WARNING: Data will not be log transformed and scaled')
        print('         Columns with zero summed columns will not be removed')
        print('         To change this behavior, set scale_data to True')

    return adata



