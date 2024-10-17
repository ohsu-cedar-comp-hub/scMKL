import tracemalloc
import numpy as np
import scipy
import anndata as ad
import gc

from scmkl.tfidf import tfidf_filter


def filter_features(X, feature_names, group_dict):
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

    # Store all objects in dictionary in array
    for group in group_dict.keys():
        group_features.update(set(group_dict[group]))

        group_dict[group] = np.sort(np.array(list(feature_set.intersection(set(group_dict[group])))))

    # Find location of desired features in whole feature set
    group_feature_indices = np.where(np.in1d(feature_names, np.array(list(group_features)), assume_unique = True))[0]

    # Subset only the desired features and their data
    X = X[:,group_feature_indices]
    feature_names = np.array(list(feature_names))[group_feature_indices]

    return X, feature_names, group_dict


def train_test_split(y, train_indices = None, seed_obj = np.random.default_rng(100), train_ratio = 0.8):
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
    if train_indices == None:

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

def sparse_var(X, axis = None):

    '''
    Function to calculate variance on a sparse matrix.
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

def process_data(X_train, X_test = None, data_type = 'counts', return_dense = True):


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

    # Remove features that have no variance in the training data (will be uniformative)
    if data_type not in ['counts', 'binary']:
        print('Data will not be normalized for gene expression data')
        print('Columns with zero summed columns will not be removed')
        print('To change this behavior, set data_type to counts or binary')

    if X_test == None:
            X_test = X_train[:1,:] # Creates dummy matrix to for the sake of calculation without increasing computational time
            orig_test = None
    else:
        orig_test = 'given'

    var = sparse_var(X_train, axis = 0)
    variable_features = np.where(var > 1e-5)[0]

    X_train = X_train[:,variable_features]
    X_test = X_test[:, variable_features]

    #Data processing according to data type
    if data_type.lower() == 'counts':

        if scipy.sparse.issparse(X_train):
            X_train = X_train.log1p()
            X_test = X_test.log1p()
        else:
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)
            
        #Center and scale count data
        train_means = np.mean(X_train, 0)
        train_sds = np.sqrt(var[variable_features])

        X_train = (X_train - train_means) / train_sds
        X_test = (X_test - train_means) / train_sds
    
    elif data_type.lower() == 'binary':

        # TFIDF filter binary peaks
        non_empty_row = np.where(np.sum(X_train, axis = 1) > 0)[0]

        if scipy.sparse.issparse(X_train):
            non_0_cols = tfidf_filter(X_train.toarray()[non_empty_row,:], mode= 'filter')
        else:
            non_0_cols = tfidf_filter(X_train[non_empty_row,:], mode = 'filter')

        X_train = X_train[:, non_0_cols]
        X_test = X_test[:, non_0_cols]

    if return_dense and scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    if orig_test == None:
        return X_train
    else:
        return X_train, X_test


def create_adata(X, feature_names: np.ndarray, cell_labels: np.ndarray, group_dict: dict, data_type: str, split_data = None, D = 100, 
                 remove_features = False, distance_metric = 'euclidean', kernel_type = 'Gaussian', random_state = 1):
    
    '''
    Function to create an AnnData object to carry all relevant information going forward

    Input:
        X- A data matrix of cells by features can be a numpy array, scipy sparse array or pandas dataframe (sparse array recommended for large datasets)
        feature_names- A numpy array of feature names corresponding with the features in X
        cell_labels- A numpy array of cell phenotypes corresponding with the cells in X.  Must be binary
        group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
        data_type- 'counts' or 'binary'.  Determines what preprocessing is applied to the data. 
            Log transforms and standard scales counts data
            TFIDF filters binary data
        split_data- Either numpy array of precalculated train/test split for the cells -or-
                            None.  If None, the train test split will be calculated with balanced classes.
        D- Number of Random Fourier Features used to calculate Z. Should be a positive integer.
                Higher values of D will increase classification accuracy at the cost of computation time
        remove_features- Bool whether to filter the features from the dataset.
                Will remove features from X, feature_names not in group_dict and remove features from groupings not in feature_names
        random_state- Integer random_state used to set the seed for reproducibilty.
    Output:
        AnnData object with: 
            Data equal to the values in X- accessible with adata.X
            Variable names equal to the values in feature_names- accessible with adata.var_names
            Cell phenotypes equal to the values in cell_labels- accessible with adata.obs['labels']
            Train/test split either as given or calculated in this function- accessible with adata.uns['train_indices'] and adata.uns['test_indices'] respectively
            Grouping information equal to given group_dict- accessible with adata.uns['group_dict']
            seed_obj with seed equal to 100 * random_state- accessible with adata.uns['seed_obj']
            D- accessible with adata.uns['D']
            Type of data to determine preprocessing steps- accessible with adata.uns['data_type']

    '''

    assert X.shape[0] == len(cell_labels), 'Different number of cells than labels'
    assert X.shape[1] == len(feature_names), 'Different number of features in X than feature names'
    assert len(np.unique(cell_labels)) == 2, 'cell_labels must contain 2 classes'
    assert isinstance(D, int) and D > 0, 'D must be a positive integer'
    assert kernel_type.lower() in ['gaussian', 'laplacian', 'cauchy'], 'Given kernel type not implemented. Gaussian, Laplacian, and Cauchy are the acceptable types.'

    if remove_features:
        X, feature_names, group_dict = filter_features(X, feature_names, group_dict)

    adata = ad.AnnData(X)
    adata.var_names = feature_names
    adata.obs['labels'] = cell_labels
    adata.uns['group_dict'] = group_dict
    adata.uns['seed_obj'] = np.random.default_rng(100 * random_state)
    adata.uns['data_type'] = data_type
    adata.uns['D'] = D
    adata.uns['kernel_type'] = kernel_type
    adata.uns['distance_metric'] = distance_metric

    if split_data == None:
        train_indices, test_indices = train_test_split(cell_labels, seed_obj = adata.uns['seed_obj'])
    else:
        train_indices = np.where(split_data == 'train')[0]
        test_indices = np.where(split_data == 'test')[0]

    adata.uns['train_indices'] = train_indices
    adata.uns['test_indices'] = test_indices

    return adata



