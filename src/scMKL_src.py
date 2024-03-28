import numpy as np
import scipy
import celer
import sklearn

def Predict(model, X_test, y_test, metrics = None):
    '''
    Function to return predicted labels and calculate any of AUROC, Accuracy, F1 Score, Precision, Recall for a classification. 
    Input:  
            Trained model- should work with an sklearn based ML model.
            X_test- the Matrix containing the testing data (will be approximate Z in this workflow).
            y_test- Corresponding labels for testing data.  Needs to binary, but not necessarily discrete.
            metrics- Which metrics to calculate on the predicted values.
                     Only metrics for binary classification at the moment. May be extended for multiclass.

    Output:
            Values predicted by the model
            Dictionary containing AUROC, Accuracy, F1 Score, Precision, and Recall

    '''
    y_test = y_test.ravel()
    assert X_test.shape[0] == len(y_test), 'X and y must have the same number of samples'
    assert all([metric in ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall'] for metric in metrics]), 'Unknown metric provided.  Must be one or more of AUROC, Accuracy, F1-Score, Precision, Recall'

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-model.predict(X_test)))

    # Group Lasso requires 'continous' y values need to re-descritize it
    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1

    metric_dict = {}

    #Convert numerical probabilies into binary phenotype
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

def Calculate_AUROC(model, X_test, y_test)-> float:
    '''
    Function to calculate the AUROC for a classification. 
    Input:  
            Trained model- should work with an sklearn based ML model.
            X_test- the Matrix containing the testing data (will be approximate Z in this workflow).
            y_test- Corresponding labels for testing data.  Needs to binary, but not necessarily discrete.
    Output:
            Calculated AUROC value
    '''
    y_test = y_test.ravel()
    assert X_test.shape[0] == len(y_test), 'X and y must have the same number of samples'
    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-model.predict(X_test)))
    # Group Lasso requires 'continous' y values need to re-descritize it
    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, probabilities)
    auc = sklearn.metrics.auc(fpr, tpr)
    
    return(auc)

def Calculate_Z(X_train, X_test, group_dict: dict, assay: str, D: int, feature_set, sigma_list, kernel_type = 'Gaussian', seed_obj = np.random.default_rng(100)) -> tuple:
    '''
    Function to calculate approximate kernels.
    Input:
            X_train- Matrix containing training data of cells x features.  Can be scipy sparse array or numpy array
            X_test- Matrix containing testing data. Can be scipy sparse array or numpy array
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
            assay- What type of sequencing data.  Used to determine how to process the data.
                        If not rna, atac, or gene_scores, no preprocessing will be done. Will likely be updated for new modalities.
            feature_set- Numpy array containing the names of all features
            sigma_array- Numpy array with 'kernel widths' used for calculating the distribution for projection.
            kernel_type- String to determine which kernel function to approximate. Currently on Gaussian, Laplacian, and Cauchy are supported.
            seed_obj- Numpy random state used for random processes. Can be specified for reproducubility or set by default.
    Output:
            Z_train- Approximate kernel for training of shape N x 2*D*G for N training samples, D Random Fourier Features, and G groups
            Z_test- Approximate kernel for testing of shape M x 2*D*G for M training samples, D Random Fourier Features, and G groups

    '''
    assert kernel_type.lower() in ['gaussian', 'cauchy', 'laplacian'], 'Kernel function must be Gaussian, Cauchy, or Laplacian'
    assert X_train.shape[1] == len(feature_set), 'Given features do not correspond with features in X'

    #Number of groupings taking from group_dict
    N_pathway = len(group_dict.keys())

    # Create Arrays to store concatenated group Z.  Each group of features will have a corresponding entry in each array
    Z_train = np.zeros((X_train.shape[0], 2 * D * N_pathway))
    Z_test = np.zeros((X_test.shape[0], 2 * D * N_pathway))

    # Loop over each of the groups and creating Z for each
    for m, group_name in enumerate(group_dict.keys()):
        
        #Extract features from mth group
        group_features = group_dict[group_name]

        #Find indices of group features in overall array
        feature_indices = np.nonzero(np.in1d(feature_set, np.array(list(group_features))))[0]

        # Create data arrays containing only features within this group
        train_features = X_train[0:, feature_indices]
        test_features = X_test[0:, feature_indices]

        # Converts data to dense array
        if scipy.sparse.issparse(train_features):
            train_features = train_features.toarray()
            test_features = test_features.toarray()

        # Removes non-variable features
        sds = np.std(train_features, axis = 0)
        variable_features = np.where(sds > 1e-10)[0]
        
        train_features = train_features[:,variable_features]
        test_features = test_features[:, variable_features]

        #Data processing- may be controlled by a flag in the future.
        if assay.lower() in ['rna', 'gene_scores']:

            train_features = np.log1p(train_features)
            test_features = np.log1p(test_features)
            
            #Center and scale count data
            train_means = np.mean(train_features, 0)
            train_sds = np.std(train_features, 0)

            train_features = (train_features - train_means) / train_sds
            test_features = (test_features - train_means) / train_sds
        
        elif assay.lower() == 'atac':

            # TFIDF filter binary peaks
            non_empty_row = np.where(np.count_nonzero(train_features, 1) != 0)[0]

            non_0_cols = TF_IDF_filter(train_features[non_empty_row,:], mode= 'filter')

            train_features = train_features[:, non_0_cols]
            test_features = test_features[:, non_0_cols]

        #Extract pre-calculated sigma used for approximating kernel
        adjusted_sigma = sigma_list[m]

        #Calculates approximate kernel according to chosen kernel function- may add more functions in the future
        #Distribution data comes from Fourier Transform of original kernel function
        if kernel_type.lower() == 'gaussian':

            gamma = 1/(2*adjusted_sigma**2)
            sigma_p = 0.5*np.sqrt(2*gamma)

            W = seed_obj.normal(0, sigma_p, train_features.shape[1]*D).reshape((train_features.shape[1]),D)

        elif kernel_type.lower() == 'laplacian':

            gamma = 1/(2*adjusted_sigma)

            W = gamma * seed_obj.standard_cauchy(train_features.shape[1]*D).reshape((train_features.shape[1],D))

        elif kernel_type.lower() == 'cauchy':

            gamma = 1/(2*adjusted_sigma**2)
            b = 0.5*np.sqrt(gamma)

            W = seed_obj.laplace(0, b, train_features.shape[1]*D).reshape((train_features.shape[1],D))

        train_projection = np.matmul(train_features, W)
        test_projection = np.matmul(test_features, W)
        
        #Store group Z in whole-Z object.  Preserves order to be able to extract meaningful groups
        Z_train[0:, np.arange( m * 2 * D , (m + 1) * 2 * D)] = np.sqrt(1/D)*np.hstack((np.cos(train_projection), np.sin(train_projection)))
        Z_test[0:, np.arange( m * 2 * D , (m + 1) * 2 * D)] = np.sqrt(1/D)*np.hstack((np.cos(test_projection), np.sin(test_projection)))

    return Z_train, Z_test

def Estimate_Sigma(X, group_dict, assay, feature_set, distance_metric = 'euclidean', seed_obj = np.random.default_rng(100)) -> np.ndarray:
    '''
    Function to calculate approximate kernels weights to inform distribution for project of Fourier Features. Calculates one sigma per group of features
    Input:
            X- Matrix containing training data of cells x features.  Can be scipy sparse array or numpy array
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
            assay- What type of sequencing data.  Used to determine how to process the data.
                        If not rna, atac, or gene_scores, no preprocessing will be done. Will likely be updated for new modalities.
            feature_set- Numpy array containing the names of all features
            distance_metric- Pairwise distance metric to use. Must be from the list offered in scipy cdist function or a custom distance function.
            seed_obj- Numpy random state used for random processes. Can be specified for reproducubility or set by default.
    Output:
            sigma_list- Numpy array of kernel widths in the order of the groups in the group_dict to be used in Calculate_Z() function.

    '''
    assert X.shape[1] == len(feature_set), 'Given features do not correspond with features in X'
 
    sigma_list = []

    # Loop over every group
    for group_name in group_dict.keys():

        # Select only features within that group
        group_features = group_dict[group_name] 
        feature_indices = np.nonzero(np.in1d(feature_set, np.array(list(group_features))))[0]

        train_features = X[0:, feature_indices]

        if scipy.sparse.issparse(train_features):
            train_features = train_features.toarray()

        # Remove all 0 and non-variable columns
        sds = np.std(train_features, axis = 0)
        variable_features = np.where(sds > 1e-10)[0]
        
        train_features = train_features[:,variable_features]

        # Process data according to modality
        if assay.lower() in ['rna', 'gene_scores']:

            train_features = np.log1p(train_features)

            train_means = np.mean(train_features, 0)
            train_sds = np.std(train_features, 0)

            train_features = (train_features - train_means) / train_sds
        
        elif assay.lower() == 'atac':

            non_empty_row = np.where(np.count_nonzero(train_features, 1) != 0)[0]
            non_0_cols = TF_IDF_filter(train_features[non_empty_row,:])

            train_features = train_features[:, non_0_cols]

        # Sample cells because distance calculation are costly and can be approximated
        distance_indices = seed_obj.choice(np.arange(train_features.shape[0]), np.min((2000, train_features.shape[0])))

        # Calculate Distance Matrix with specified metric
        sigma = np.mean(scipy.spatial.distance.cdist(train_features[distance_indices,:], train_features[distance_indices,:], distance_metric))

        if sigma == 0:
            sigma += 1e-5

        sigma_list.append(sigma)
        
    return np.array(sigma_list)

def Optimize_Sigma(X, y, group_dict, assay, D, feature_set, sigma_list, kernel_type = 'Gaussian', seed_obj = np.random.default_rng(100), 
                   alpha = 1.9, sigma_adjustments = np.arange(0.1,2.1,0.3), k = 4) -> np.ndarray:
    '''
    Function to perform k-fold cross-validation to optimize sigma (kernel widths) based on classification AUROC
    Inputs:
            X- Matrix containing data that will be split into training and testing data for cross validation.
            Best practice is to not include any data from the testing set.
            y- Samples labels corresponding to X.
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
            assay- What type of sequencing data.  Used to determine how to process the data.
                        If not rna, atac, or gene_scores, no preprocessing will be done. Will likely be updated for new modalities.
            feature_set- Numpy array containing the names of all features
            sigma_array- Numpy array with 'kernel widths' used for calculating the distribution for projection.
            kernel_type- String to determine which kernel function to approximate. Currently on Gaussian, Laplacian, and Cauchy are supported.
            seed_obj- Numpy random state used for random processes. Can be specified for reproducubility or set by default.
            alpha- Group Lasso regularization coefficient. alpha is a floating point value controlling model solution sparsity. Must be a positive float.
                        The smaller the value, the more feature groups will be selected in the trained model.
            sigma_adjustments- Iterable containing values to multiply sigmas by. Must be positive values.
            k- Number of cross validation steps. Use k = # Samples for leave-one-out cross validation. Must be a positive integer.
    Outputs:
            optimized_sigma- Numpy array with sigma array producing highest AUROC classification
    '''

    assert np.all(sigma_list > 0), 'Kernel Widths must be positive'
    assert np.all(sigma_adjustments > 0), 'Adjustment values must be positive'
    assert X.shape[0] == len(y), 'X and y must have the same number of samples'
    assert X.shape[1] == len(feature_set), 'Given features do not correspond with features in X'
    assert kernel_type.lower() in ['gaussian', 'cauchy', 'laplacian'], 'Kernel function must be Gaussian, Cauchy, or Laplacian'
    assert isinstance(k, int) and k > 0, 'Must be a positive integer number of folds'
    
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)

    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k
    sigma_list = np.array(sigma_list)

    auc_array = np.zeros((len(sigma_adjustments), k))

    for fold in np.arange(k):
        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))

        X_train = X[fold_train,:]
        X_test = X[fold_test,:]
        y_train = y[fold_train]
        y_test = y[fold_test]

        for i, adj in enumerate(sigma_adjustments):
            new_sigma_list = sigma_list * adj
            Z_train, Z_test = Calculate_Z(X_train, X_test, group_dict, assay, D, feature_set, new_sigma_list, kernel_type, seed_obj)

            gl = Train_Model(Z_train, y_train, group_size= 2 * D, alpha = alpha)
            auc_array[i, fold] = Calculate_AUROC(gl, Z_test, y_test)
    
    best_adj = sigma_adjustments[np.argmax(np.mean(auc_array, axis = 1))]
    optimized_sigma = sigma_list * best_adj

    return optimized_sigma

def Train_Model(X_train, y_train, group_size = 1, alpha = 0.9) -> celer.dropin_sklearn.GroupLasso:
    '''
    Function to train and test grouplasso model
    Inputs:
            X_train- Training Data of samples by features.  Must be a numpy array
            y_train- Sample labels corresponding to training data. Must be binary
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
            model- The trained Celer Group Lasso model.  Used in Find_Selected_Pathways() function for interpretability.
                    For more information about attributes and methods see the Celer documentation at https://mathurinm.github.io/celer/generated/celer.GroupLasso.html.

    '''
    assert X_train.shape[0] == len(y_train), 'X and y must have the same number of samples'

    cell_labels = np.unique(y_train)

    assert len(cell_labels) == 2, f'Sample labels must be binary for this algorithm. Expected 2 unique labels, input has {len(cell_labels)}.'

    # This is a regression algorithm. We need to make the labels 'continuous' for classification, but they will remain binary.
    train_labels = np.ones(y_train.shape)
    train_labels[y_train == cell_labels[1]] = -1

    # train_labels = (train_labels - np.mean(train_labels)) / np.std(train_labels)

    # Alphamax is an attempt to regularize the effect of alpha (a sparsity parameter) across different data sets
    alphamax = np.max(np.abs(X_train.T.dot(train_labels))) / X_train.shape[0] * alpha

    # Instantiate celer Group Lasso Regression Model Object
    model = celer.GroupLasso(groups = group_size, alpha = alphamax)

    # Fit model using training data
    model.fit(X_train, train_labels.ravel())

    return model

def Optimize_Alpha(X_train, y_train, group_size, starting_alpha = 1.9, increment = 0.2, target = 1, n_iter = 10):
    '''
    Iteratively train a grouplasso model and update alpha to find the parameter yielding the desired sparsity.
    This function is meant to find a good starting point for your model, and the alpha may need further fine tuning.
    Input:
        X_train- Matrix containing data that will be split into training and testing data for cross validation.
            Best practice is to not include any data from the testing set.
        y_train- Samples labels corresponding to X.
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
    assert X_train.shape[0] == len(y_train), 'X and y must have the same number of samples'
    assert increment > 0 and increment < starting_alpha, 'Choose a positive increment less than alpha'
    assert target > 0 and isinstance(target, int), 'Choose an integer target number of groups that is greater than 0'
    assert n_iter > 0 and isinstance(n_iter, int), 'Choose an integer number of iterations that is greater than 0'

    if isinstance(group_size, int):
        num_groups = int(X_train.shape[1]/group_size)
    else:
        num_groups = len(group_size)

    sparsity_dict = {}
    alpha = starting_alpha

    for i in np.arange(n_iter):
        model = Train_Model(X_train, y_train, group_size, alpha)
        num_selected = len(Find_Selected_Pathways(model, np.arange(num_groups)))

        sparsity_dict[np.round(alpha,4)] = num_selected

        if num_selected < target:
            #Decreasing alpha will increase the number of selected pathways
            if alpha - increment in sparsity_dict.keys():
                # Make increment smaller so the model can't go back and forth between alpha values
                increment /= 2
            alpha = np.max([alpha - increment, 1e-3]) #Ensures that alpha will never be negative
        elif num_selected > target:
            if alpha + increment in sparsity_dict.keys():
                increment /= 2
            alpha += increment
        if num_selected == target:
            break

    optimal_alpha = list(sparsity_dict.keys())[np.argmin([np.abs(selected - target) for selected in sparsity_dict.values()])]
    return sparsity_dict, optimal_alpha

def Find_Selected_Pathways(model, group_names) -> np.ndarray:
    '''
    Function to find feature groups selected by the model during training.  If feature weight assigned by the model is non-0, then the group containing that feature is selected.
    Inputs:
        model- A trained celer.GroupLasso model.
        group_names- An iterable object containing the group_names in the same order as the feature groupings from Data array.
    Outpus:
        Numpy array containing the names of the groups selected by the model.
    '''

    selected_groups = []
    coefficients = model.coef_
    group_size = model.get_params()['groups']


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
    Function to use Term Frequency Inverse Document Frequency filtering for atac data to find meaningful features (Needs to be optimized). 
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
        X = X.toarray()
    
    doc_freq = np.array(np.sum(X > 0, axis=0)).flatten()
    idf = np.log(X.shape[0] / (1 + doc_freq))
    tf = X * 1 / np.sum(X, axis = 1, keepdims= True)
    tfidf = tf * idf

    if mode == 'normalize':
        return tfidf
    if mode == 'filter':
        return np.where(np.sum(tfidf, axis = 0) > 0)[0]

def Filter_Features(X, feature_names, group_dict):
    '''
    Function to remove unused features from X matrix.  Any features not included in group_dict will be removed from the matrix.
    Input:
            X- Data array. Can be Numpy array or Scipy Sparse Array
            feature_names- Numpy array of corresponding feature names
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
    Output:
            X- Data array containing data only for features in the group_dict
            feature_names- Numpy array of corresponding feature names from group_dict
    '''

    group_features = set()

    # Store all objects in dictionary in array
    for group in group_dict.keys():
        group_features.update(set(group_dict[group]))

    # Find location of desired features in whole feature set
    group_feature_indices = np.nonzero(np.isin(feature_names, np.array(list(group_features))))[0]

    # Subset only the desired features and their data
    X = X[:,group_feature_indices]
    feature_names = feature_names[group_feature_indices]

    return X, feature_names

def Combine_Modalities(Assay_1_name: str, Assay_2_name: str,
                       Assay_1_Z_train, Assay_2_Z_train, 
                       Assay_1_Group_Names, Assay_2_Group_Names,
                       Assay_1_Z_test = np.zeros((0)), Assay_2_Z_test = np.zeros((0)),) -> tuple:
    '''
    Combines data sets for multimodal classification.  Combined group names are assay+group_name
    Input:
            Assay_#_name: Name of assay to be added to group_names as a string
            Assay_#_Z_train: Numpy array containing train data
            Assay_#_Group_Names: Names of groups for the given data set
            Assay_#_Z_test: Numpy array containing test.  Is None by default 
    Output:
            combined_Z: Concatenated data matrices as numpy array
            combined_group_names = Concatenated group names of form assay+group_name
    '''
    assert Assay_1_Z_train.shape[0] == Assay_2_Z_train.shape[0] and Assay_1_Z_test.shape[0] == Assay_2_Z_test.shape[0], 'Cannot concatenate arrays with different number of rows.'

    combined_Z_train = np.hstack((Assay_1_Z_train, Assay_2_Z_train))

    Assay_1_Group_Names = [f'{Assay_1_name}_{name}' for name in Assay_1_Group_Names]
    Assay_2_Group_Names = [f'{Assay_2_name}_{name}' for name in Assay_2_Group_Names]

    combined_group_names = np.concatenate((Assay_1_Group_Names, Assay_2_Group_Names))

    if Assay_1_Z_test.shape[0] == 0 or Assay_2_Z_test.shape[0] == 0:
        return combined_Z_train, combined_group_names
    else:
        combined_Z_test = np.hstack((Assay_1_Z_test, Assay_2_Z_test))
        return combined_Z_train, combined_Z_test, combined_group_names

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
    return var