import tracemalloc
import numpy as np
import scipy
import sklearn
import anndata as ad
import sys

def Predict(adata, metrics = None):
    '''
    Function to return predicted labels and calculate any of AUROC, Accuracy, F1 Score, Precision, Recall for a classification. 
    Input:  
            adata- adata object with trained model and Z matrices in uns
            metrics- Which metrics to calculate on the predicted values

    Output:
            Values predicted by the model
            Dictionary containing AUROC, Accuracy, F1 Score, Precision, and/or Recall depending on metrics argument

    '''
    y_test = adata.obs['labels'].iloc[adata.uns['test_indices']].to_numpy()
    X_test = adata.uns['Z_test']
    assert X_test.shape[0] == len(y_test), 'X and y must have the same number of samples'
    assert all([metric in ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall'] for metric in metrics]), 'Unknown metric provided.  Must be one or more of AUROC, Accuracy, F1-Score, Precision, Recall'

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-adata.uns['model'].predict(X_test)))

    # Group Lasso requires 'continous' y values need to re-descritize it
    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1

    metric_dict = {}

    #Convert numerical probabilities into binary phenotype
    y_pred = np.array(np.repeat(np.unique(y_test)[1], len(y_test)), dtype = 'object')
    y_pred[np.round(probabilities,0).astype(int) == 1] = np.unique(y_test)[0]

    if metrics == None:
        return y_pred

    if 'AUROC' in metrics:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y, probabilities)
        metric_dict['AUROC'] = sklearn.metrics.auc(fpr, tpr)
    if 'Accuracy' in metrics:
        metric_dict['Accuracy'] = np.mean(y_test == y_pred)
    if 'F1-Score' in metrics:
        metric_dict['F1-Score'] = sklearn.metrics.f1_score(y_test, y_pred, pos_label = np.unique(y_test)[0])
    if 'Precision' in metrics:
        metric_dict['Precision'] = sklearn.metrics.precision_score(y_test, y_pred, pos_label = np.unique(y_test)[0])
    if 'Recall' in metrics:
        metric_dict['Recall'] = sklearn.metrics.recall_score(y_test, y_pred, pos_label = np.unique(y_test)[0])

    return y_pred, metric_dict

def Calculate_AUROC(adata)-> float:
    '''
    Function to calculate the AUROC for a classification. 
    Designed as a helper function.  Recommended to use Predict() for model evaluation.
    Input:  
            adata- adata object with trained model and Z matrices in uns
    Output:
            Calculated AUROC value
    '''

    y_test = adata.obs['labels'].iloc[adata.uns['test_indices']].to_numpy()
    X_test = adata.uns['Z_test']

    y_test = y_test.ravel()
    assert X_test.shape[0] == len(y_test), f'X has {X_test.shape[0]} samples and y has {len(y_test)} samples.'

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-adata.uns['model'].predict(X_test)))
    # Group Lasso requires 'continous' y values need to re-descritize it

    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, probabilities)
    auc = sklearn.metrics.auc(fpr, tpr)
    
    return(auc)

def Calculate_Z(adata, kernel_type = 'Gaussian', n_features = 5000) -> tuple:
    '''
    Function to calculate approximate kernels.
    Input:
            adata- Adata obj as created by `Create_Adata`
            Sigma can be calculated with Estimate_Sigma or a heuristic but must be positive.
            kernel_type- String to determine which kernel function to approximate. Currently only Gaussian, Laplacian, and Cauchy are supported.
            n_features- Number of random feature to use when calculating Z- used for scalability
    Output:
            adata_object with Z matrices accessible with- adata.uns['Z_train'] and adata.uns['Z_test'] respectively

    '''
    assert kernel_type.lower() in ['gaussian', 'cauchy', 'laplacian'], 'Kernel function must be Gaussian, Cauchy, or Laplacian'
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


        X_train, X_test = Process_Data(X_train = X_train, X_test = X_test, data_type = adata.uns['data_type'], return_dense = True)

        #Extract pre-calculated sigma used for approximating kernel
        adjusted_sigma = adata.uns['sigma'][m]

        #Calculates approximate kernel according to chosen kernel function- may add more functions in the future
        #Distribution data comes from Fourier Transform of original kernel function
        if kernel_type.lower() == 'gaussian':

            gamma = 1/(2*adjusted_sigma**2)
            sigma_p = 0.5*np.sqrt(2*gamma)

            W = adata.uns['seed_obj'].normal(0, sigma_p, X_train.shape[1]*D).reshape((X_train.shape[1]),D)

        elif kernel_type.lower() == 'laplacian':

            gamma = 1/(2*adjusted_sigma)

            W = gamma * adata.uns['seed_obj'].standard_cauchy(X_train.shape[1]*D).reshape((X_train.shape[1],D))

        elif kernel_type.lower() == 'cauchy':

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

def Estimate_Sigma(adata, distance_metric = 'euclidean', n_features = 5000):
    '''
    Function to calculate approximate kernels widths to inform distribution for project of Fourier Features. Calculates one sigma per group of features
    Input:
            adata- Adata obj as created by `Create_Adata`
            distance_metric- Pairwise distance metric to use. Must be from the list offered in scipy cdist function or a custom distance function.
            n_features- Number of random features to include when estimating sigma.  Will be scaled for the whole pathway set according to a heuristic. Used for scalability
    Output:
            adata object with sigma values.  Sigmas accessible by adata.uns['sigma']

    '''
 
    sigma_list = []

    # Loop over every group in group_dict
    for group_name, group_features in adata.uns['group_dict'].items():

        # Select only features within that group and downsample for scalability
        num_group_features = len(group_features)
        group_features = adata.uns['seed_obj'].choice(np.array(list(group_features)), min([n_features, num_group_features]), replace = False) 

        X_train = adata[adata.uns['train_indices'], group_features].X

        X_train = Process_Data(X_train = X_train, data_type = adata.uns['data_type'], return_dense = True)
        
        # Sample cells because distance calculation are costly and can be approximated
        distance_indices = adata.uns['seed_obj'].choice(np.arange(X_train.shape[0]), np.min((2000, X_train.shape[0])))

        # Calculate Distance Matrix with specified metric
        sigma = np.mean(scipy.spatial.distance.cdist(X_train[distance_indices,:], X_train[distance_indices,:], distance_metric))

        if sigma == 0:
            sigma += 1e-5

        if n_features < num_group_features:
            sigma = sigma * num_group_features / n_features # Heuristic we calculated to account for fewer features used in distance calculation

        sigma_list.append(sigma)
    
    adata.uns['sigma'] = np.array(sigma_list)
        
    return adata

def Optimize_Sigma(adata, kernel_type = 'Gaussian', alpha = 1.9, sigma_adjustments = np.arange(0.1,2.1,0.3), k = 4):
    '''
    Function to perform k-fold cross-validation to optimize sigma (kernel widths) based on classification AUROC of a validation set
    Inputs:
            adata- Adata obj as created by `Create_Adata` with sigma values. 
                    Sigma can be calculated with Estimate_Sigma or a heuristic but must be positive.
            kernel_type- String to determine which kernel function to approximate. Currently on Gaussian, Laplacian, and Cauchy are supported.
            alpha- Group Lasso regularization coefficient. alpha is a floating point value controlling model solution sparsity. Must be a positive float.
                        The smaller the value, the more feature groups will be selected in the trained model.
            sigma_adjustments- Iterable containing values to multiply sigmas by. Must be positive values.
            k- Number of cross validation steps. Use k = # Samples for leave-one-out cross validation. Must be a positive integer.
    Outputs:
            optimized_sigma- Numpy array with sigma array producing highest AUROC classification
    '''

    assert np.all(sigma_adjustments > 0), 'Adjustment values must be positive'
    assert isinstance(k, int) and k > 0, 'Must be a positive integer number of folds'
    assert np.all(adata.uns['sigma'] > 0), 'Sigma must be positive'

    import warnings 
    warnings.filterwarnings('ignore') 
    
    # Create train/validation sets with equal proportions of phenotypes

    y = adata.obs['labels'].iloc[adata.uns['train_indices']].to_numpy()
    
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)

    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k

    auc_array = np.zeros((len(sigma_adjustments), k))

    for fold in np.arange(k):
        
        print(f'Fold {fold}:\n Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')

        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))



        for i, adj in enumerate(sigma_adjustments):

            cv_adata = adata[adata.uns['train_indices'],:]
            cv_adata.uns['train_indices'] = fold_train
            cv_adata.uns['test_indices'] = fold_test

            print(f'  1. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            cv_adata.uns['sigma'] *= adj
            print(f'  2. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            cv_adata = Calculate_Z(cv_adata, kernel_type, n_features = 5000)
            print(f'  3. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            cv_adata = Train_Model(cv_adata, group_size= 2 * adata.uns['D'], alpha = alpha)
            print(f'  4. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            auc_array[i, fold] = Calculate_AUROC(cv_adata)
            print(f'  5. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}\n')

            del cv_adata

    # Take AUROC mean across the k folds
    best_adj = sigma_adjustments[np.argmax(np.mean(auc_array, axis = 1))]
    adata.uns['sigma'] *= best_adj

    print(best_adj)

    return adata

def Train_Model(adata, group_size = 1, alpha = 0.9):
    import celer
    '''
    Function to fit a grouplasso model to the provided data.
    Inputs:
            Adata with Z matrices in adata.uns
            group_size- Argument describing how the features are grouped. 
                    From Celer documentation:
                    "groupsint | list of ints | list of lists of ints.
                        Partition of features used in the penalty on w. 
                            If an int is passed, groups are contiguous blocks of features, of size groups. 
                            If a list of ints is passed, groups are assumed to be contiguous, group number g being of size groups[g]. 
                            If a list of lists of ints is passed, groups[g] contains the feature indices of the group number g."
                    If 1, model will behave identically to Lasso Regression
            alpha- Group Lasso regularization coefficient. alpha is a floating point value controlling model solution sparsity. Must be a positive float.
                        The smaller the value, the more feature groups will be selected in the trained model.
    Outputs:

            adata object with trained model in uns accessible with- adata.uns['model']
                Specifics of model:
                    model- The trained Celer Group Lasso model.  Used in Find_Selected_Pathways() function for interpretability.
                        For more information about attributes and methods see the Celer documentation at https://mathurinm.github.io/celer/generated/celer.GroupLasso.html.

    '''
    assert alpha > 0, 'Alpha must be positive'

    y_train = adata.obs['labels'].iloc[adata.uns['train_indices']]
    X_train = adata.uns['Z_train']

    cell_labels = np.unique(y_train)

    # This is a regression algorithm. We need to make the labels 'continuous' for classification, but they will remain binary.
    # Casts training labels to array of -1,1
    train_labels = np.ones(y_train.shape)
    train_labels[y_train == cell_labels[1]] = -1

    # Alphamax is a calculation to regularize the effect of alpha (a sparsity parameter) across different data sets
    alphamax = np.max(np.abs(X_train.T.dot(train_labels))) / X_train.shape[0] * alpha

    # Instantiate celer Group Lasso Regression Model Object
    model = celer.GroupLasso(groups = group_size, alpha = alphamax)

    # Fit model using training data
    model.fit(X_train, train_labels.ravel())

    adata.uns['model'] = model
    return adata

def Optimize_Sparsity(adata, group_size, starting_alpha = 1.9, increment = 0.2, target = 1, n_iter = 10):
    '''
    Iteratively train a grouplasso model and update alpha to find the parameter yielding the desired sparsity.
    This function is meant to find a good starting point for your model, and the alpha may need further fine tuning.
    Input:
        adata- Anndata object with Z_train and Z_test calculated
        group_size- Argument describing how the features are grouped. 
            From Celer documentation:
            "groupsint | list of ints | list of lists of ints.
                Partition of features used in the penalty on w. 
                    If an int is passed, groups are contiguous blocks of features, of size groups. 
                    If a list of ints is passed, groups are assumed to be contiguous, group number g being of size groups[g]. 
                    If a list of lists of ints is passed, groups[g] contains the feature indices of the group number g."
            If 1, model will behave identically to Lasso Regression.
        starting_alpha- The alpha value to start the search at.
        increment- amount to adjust alpha by between iterations
        target- The desired number of groups selected by the model.
        n_iter- The maximum number of iterations to run
            
    Output:
        sparsity_dict- Dictionary with tested alpha as keys and the number of selected pathways as the values
        alpha- The alpha value yielding the number of selected groups closest to the target.
    '''
    assert increment > 0 and increment < starting_alpha, 'Choose a positive increment less than alpha'
    assert target > 0 and isinstance(target, int), 'Choose an integer target number of groups that is greater than 0'
    assert n_iter > 0 and isinstance(n_iter, int), 'Choose an integer number of iterations that is greater than 0'

    y_train = adata.obs['labels'].iloc[adata.uns['train_indices']].to_numpy()
    X_train = adata.uns['Z_train']

    if isinstance(group_size, int):
        num_groups = int(X_train.shape[1]/group_size)
    else:
        num_groups = len(group_size)

    sparsity_dict = {}
    alpha = starting_alpha

    for i in np.arange(n_iter):
        model = Train_Model(adata, group_size, alpha)
        num_selected = len(Find_Selected_Pathways(adata))

        sparsity_dict[np.round(alpha,4)] = num_selected

        if num_selected < target:
            #Decreasing alpha will increase the number of selected pathways
            if alpha - increment in sparsity_dict.keys():
                # Make increment smaller so the model can't go back and forth between alpha values
                increment /= 2
            alpha = np.max([alpha - increment, 1e-1]) #Ensures that alpha will never be negative
        elif num_selected > target:
            if alpha + increment in sparsity_dict.keys():
                increment /= 2
            alpha += increment
        elif num_selected == target:
            break

    optimal_alpha = list(sparsity_dict.keys())[np.argmin([np.abs(selected - target) for selected in sparsity_dict.values()])]
    return sparsity_dict, optimal_alpha

def Optimize_Alpha(adata, group_size, alpha_array = np.round(np.linspace(1.9,0.1, 10),2), k = 4):
    '''
    Iteratively train a grouplasso model and update alpha to find the parameter yielding the desired sparsity.
    This function is meant to find a good starting point for your model, and the alpha may need further fine tuning.
    Input:
        adata- Anndata object with Z_train and Z_test calculated
        group_size- Argument describing how the features are grouped. 
            From Celer documentation:
            "groupsint | list of ints | list of lists of ints.
                Partition of features used in the penalty on w. 
                    If an int is passed, groups are contiguous blocks of features, of size groups. 
                    If a list of ints is passed, groups are assumed to be contiguous, group number g being of size groups[g]. 
                    If a list of lists of ints is passed, groups[g] contains the feature indices of the group number g."
            If 1, model will behave identically to Lasso Regression.
        starting_alpha- The alpha value to start the search at.
        alpha_array- Numpy array of all alpha values to be tested
        k- number of folds to perform cross validation over
            
    Output:
        sparsity_dict- Dictionary with tested alpha as keys and the number of selected pathways as the values
        alpha- The alpha value yielding the number of selected groups closest to the target.
    '''

    assert isinstance(k, int) and k > 0, 'Must be a positive integer number of folds'

    import gc
    import warnings 
    warnings.filterwarnings('ignore')

    y = adata.obs['labels'].iloc[adata.uns['train_indices']].to_numpy()
    
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)

    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k

    auc_array = np.zeros((len(alpha_array), k))

    cv_adata = adata[adata.uns['train_indices'],:]

    cv_adata.uns['train_indices'] = None
    cv_adata.uns['test_indices'] = None

    Z_train = adata.uns['Z_train'].copy()

    adata = None
    gc.collect()

    for fold in np.arange(k):
        
        print(f'Fold {fold + 1}:\n Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')

        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))

        for i, alpha in enumerate(alpha_array):

            cv_adata.uns['train_indices'] = fold_train
            cv_adata.uns['test_indices'] = fold_test
            cv_adata.uns['Z_train'] = Z_train[fold_train]
            cv_adata.uns['Z_test'] = Z_train[fold_test]


            # print(f'  1. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            # print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            cv_adata = Train_Model(cv_adata, group_size, alpha = alpha)

            # print(f'  2. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            # print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            auc_array[i, fold] = Calculate_AUROC(cv_adata)

            # print(f'  3. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            # print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')
        gc.collect()

    # Take AUROC mean across the k folds
    alpha_star = alpha_array[np.argmax(np.mean(auc_array, axis = 1))]
    cv_adata = None
    return alpha_star

def Find_Selected_Pathways(adata) -> np.ndarray:

    '''
    Function to find feature groups selected by the model during training.  If feature weight assigned by the model is non-0, then the group containing that feature is selected.
    Inputs:
        model- A trained celer.GroupLasso model.
        group_names- An iterable object containing the group_names in the same order as the feature groupings from Data array.
    Outpus:
        Numpy array containing the names of the groups selected by the model.
    '''

    selected_groups = []
    coefficients = adata.uns['model'].coef_
    group_size = adata.uns['model'].get_params()['groups']
    group_names = np.array(list(adata.uns['group_dict'].keys()))


    for i, group in enumerate(group_names):
        if not isinstance(group_size, (list, set, np.ndarray, tuple)):
            group_norm = np.linalg.norm(coefficients[np.arange(i * group_size, (i+1) * group_size - 1)])
        else: 
            group_norm = np.linalg.norm(coefficients[group_size[i]])

        if group_norm != 0:
            selected_groups.append(group)

    return np.array(selected_groups)

def TF_IDF_filter(X, mode = 'filter'):
    '''
    Function to use Term Frequency Inverse Document Frequency filtering for atac data to find meaningful features. 
    If input is pandas data frame or scipy sparse array, it will be converted to a numpy array.
    Input:
            x- Data matrix of cell x feature.  Must be a Numpy array or Scipy sparse array.
            mode- Argument to determine what to return.  Must be filter or normalize
    Output:
            TFIDF- Output depends on given 'mode' parameter
                'filter'- returns which column sums are non 0 i.e. which features are significant
                'normalize'- returns TFIDF filtered data matrix of the same dimensions as x. Returns as scipy sparse matrix
    '''

    assert mode in ['filter', 'normalize'], 'mode must be "filter" or "normalize".'
    
    if scipy.sparse.issparse(X):
        # row_sum = np.array(X.sum(axis=1)).flatten()
        tf = scipy.sparse.csc_array(X)# / row_sum[:, np.newaxis])
        doc_freq = np.array(np.sum(X > 0, axis=0)).flatten()
    else:
        # row_sum = np.sum(X, axis=1, keepdims=True)
        tf = X# / row_sum    
        doc_freq = np.sum(X > 0, axis=0)

    idf = np.log1p((1 + X.shape[0]) / (1 + doc_freq))
    tfidf = tf * idf

    if mode == 'normalize':
        if scipy.sparse.issparse(tfidf):
            tfidf = scipy.sparse.csc_matrix(tfidf)
        return tfidf
    elif mode == 'filter':
        significant_features = np.where(np.sum(tfidf, axis=0) > 0)[0]
        return significant_features

def TF_IDF_normalize(adata, binarize = False):

    '''
    Function to TF IDF normalize the data in an adata object
    If train/test indices are included in the object, it will calculate the normalization separately for the training and testing data
        Otherwise it will calculate it on the entire dataset
    If any rows are entirely 0, that row and its metadata will be removed from the object

    Input:
        adata- adata object with data in adata.X to be normalized
            Can have train/test indices included or not
        binarize- Boolean option to binarize the data
    Output:
        adata- adata object with same attributes as before, but the TF IDF normalized matrix in place of adata.X
                    Will now have the train data stacked on test data, and the indices will be adjusted accordingly
    '''

    row_sums = np.sum(adata.X, axis = 1)
    X = adata[np.where(row_sums > 0)[0],:].X.copy()

    if binarize:
        X[X>0] = 1

    if 'train_indices' in adata.uns_keys():

        train_indices = adata.uns['train_indices']
        test_indices = adata.uns['test_indices']

        del adata.uns['train_indices'], adata.uns['test_indices']
        
        train_indices = train_indices[np.where(train_indices < X.shape[0])]
        test_indices = test_indices[np.where(test_indices < X.shape[0])]

        tfidf_train = TF_IDF_filter(X[train_indices,:], mode = 'normalize')
        tfidf_test = TF_IDF_filter(X[test_indices,:], mode = 'normalize')

        if scipy.sparse.issparse(adata.X):
            tfidf_norm = scipy.sparse.vstack((tfidf_train, tfidf_test))
        else:
            tfidf_norm = np.vstack((tfidf_train, tfidf_test))

        labels = np.concatenate((adata.obs['labels'][train_indices], adata.obs['labels'][test_indices]))
        train_indices = np.arange(len(train_indices))
        test_indices = np.arange(len(train_indices), len(train_indices) + len(test_indices))

        adata.uns['train_indices'] = train_indices
        adata.uns['test_indices'] = test_indices
    else:
        tfidf_norm = TF_IDF_filter(adata.X, mode = 'normalize')
        labels = new_adata.obs['labels']

    new_adata = ad.AnnData(tfidf_norm)
    new_adata.obs['labels'] = labels
    new_adata.var_names = adata.var_names.copy()
    new_adata.uns = adata.uns.copy()

    return new_adata

def Filter_Features(X, feature_names, group_dict):
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

def Combine_Modalities(Assay_1_name: str, Assay_2_name: str,
                       Assay_1_adata, Assay_2_adata,
                       combination = 'concatenate'):
    '''
    Combines data sets for multimodal classification.  Combined group names are assay+group_name
    Input:
            Assay_#_name: Name of assay to be added to group_names as a string if overlap
            Assay_#_adata: Anndata object containing Z matrices and annotations
            combination: How to combine the matrices, either sum or concatenate
    Output:
            combined_adata: Adata object with the combined Z matrices and annotations.  Annotations will be assumed to match
    '''
    assert Assay_1_adata.shape[0] == Assay_2_adata.shape[0], 'Cannot combine data with different number of cells.'
    assert Assay_1_name != Assay_2_name, 'Assay names must be distinct'
    assert combination.lower() in ['sum', 'concatenate']

    assay1_groups = set(list(Assay_1_adata.uns['group_dict'].keys()))
    assay2_groups = set(list(Assay_2_adata.uns['group_dict'].keys()))

    combined_adata = ad.AnnData(obs = Assay_1_adata.obs, uns = Assay_1_adata.uns)

    if combination == 'concatenate':
        combined_adata.uns['Z_train'] = np.hstack((Assay_1_adata.uns['Z_train'], Assay_2_adata.uns['Z_train']))
        combined_adata.uns['Z_test'] = np.hstack((Assay_1_adata.uns['Z_test'], Assay_2_adata.uns['Z_test']))

    elif combination == 'sum':
        assert Assay_1_adata.uns['Z_train'].shape == Assay_2_adata.uns['Z_train'].shape, 'Cannot sum Z matrices with different dimensions'
        combined_adata.uns['Z_train'] = Assay_1_adata.uns['Z_train'] + Assay_2_adata.uns['Z_train']
        combined_adata.uns['Z_test'] = Assay_1_adata.uns['Z_test'] + Assay_2_adata.uns['Z_test']

    group_dict1 = Assay_1_adata.uns['group_dict']
    group_dict2 = Assay_2_adata.uns['group_dict']

    if len(assay1_groups.intersection(assay2_groups)) > 0:
        new_dict = {}
        for group, features in group_dict1.items():
            new_dict[f'{Assay_1_name}-{group}'] = features
    
        group_dict1 = new_dict

        new_dict = {}
        for group, features in group_dict2.items():
            new_dict[f'{Assay_2_name}-{group}'] = features
    
        group_dict2 = new_dict

    group_dict = group_dict1 | group_dict2 #Combines the dictionaries
    combined_adata.uns['group_dict'] = group_dict

    return combined_adata

def Train_Test_Split(y, train_indices = None, seed_obj = np.random.default_rng(100), train_ratio = 0.8):
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

def Sparse_Var(X, axis = None):

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

def Process_Data(X_train, X_test = None, data_type = 'counts', return_dense = True):


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
    assert data_type in ['counts', 'binary'], 'Improper value given for data_type'

    if X_test == None:
            X_test = X_train[:1,:] # Creates dummy matrix to for the sake of calculation without increasing computational time
            orig_test = None
    else:
        orig_test = 'given'

    var = Sparse_Var(X_train, axis = 0)
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
            non_0_cols = TF_IDF_filter(X_train.toarray()[non_empty_row,:], mode= 'filter')
        else:
            non_0_cols = TF_IDF_filter(X_train[non_empty_row,:], mode = 'filter')

        X_train = X_train[:, non_0_cols]
        X_test = X_test[:, non_0_cols]

    if return_dense and scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    if orig_test == None:
        return X_train
    else:
        return X_train, X_test

def Create_Adata(X, feature_names: np.ndarray, cell_labels: np.ndarray, group_dict: dict, data_type: str, train_test_split = None, D = 100, filter_features = False, random_state = 1):
    
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
        train_test_split- Either numpy array of precalculated train/test split for the cells -or-
                            None.  If None, the train test split will be calculated with balanced classes.
        D- Number of Random Fourier Features used to calculate Z. Should be a positive integer.
                Higher values of D will increase classification accuracy at the cost of computation time
        filter_features- Bool whether to filter the features from the dataset.
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
    assert data_type in ['counts', 'binary'], 'data_type must be either "counts" or "binary"'
    assert isinstance(D, int) and D > 0, 'D must be a positive integer'

    if filter_features:
        X, feature_names, group_dict = Filter_Features(X, feature_names, group_dict)

    adata = ad.AnnData(X)
    adata.var_names = feature_names
    adata.obs['labels'] = cell_labels
    adata.uns['group_dict'] = group_dict
    adata.uns['seed_obj'] = np.random.default_rng(100 * random_state)
    adata.uns['data_type'] = data_type
    adata.uns['D'] = D

    if train_test_split == None:
        train_indices, test_indices = Train_Test_Split(cell_labels, seed_obj = adata.uns['seed_obj'])
    else:
        train_indices = np.where(train_test_split == 'train')[0]
        test_indices = np.where(train_test_split == 'test')[0]

    adata.uns['train_indices'] = train_indices
    adata.uns['test_indices'] = test_indices


    return adata