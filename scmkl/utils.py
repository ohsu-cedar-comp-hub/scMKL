import numpy as np
import anndata as ad
import time
import tracemalloc
import gc

from scmkl.train import train_model
from scmkl.test import predict, find_selected_groups
from scmkl.estimate_sigma import estimate_sigma
from scmkl.approx_kernels import calculate_z
from scmkl.multimodal_processing import multimodal_processing


def _check_adatas(adatas : list, check_uns : bool = False, 
                  check_obs : bool = False) -> None:
    '''
    Takes a list of AnnData objects and checks that all training and
    testing indices stored in adata.uns and cell labels stored in 
    adata.obs are the same. Ensures that all objects are of type 
    AnnData.
    Args:
        adatas - a list of AnnData objects where each element has the
            same array of indices for adata.uns['train_indices'] and 
            the same array of indices for adata.uns-'test_indices'].
            Additionally, adata.obs['labels'] must be the same for all
            AnnData objects.
        check_uns - a boolean value. If True, function will ensure that
            'train_indices' and 'test_indices' in adata.uns are the
            same. If False, indices will not be checked.
        check_obs - a boolean value. If True, function will ensure that
            adata.obs['labels'] in each AnnData object are the same.
    Returns:
        Returns None. Will throw an error if not all of the criteria
        listed above are met by adatas.
    '''
    for i, adata in enumerate(adatas):
        # Ensuring all elements are type AnnData
        if type(adata) != ad.AnnData: 
            raise TypeError(f'List object in position {i} is {type(adata)}, \
                            all elements in adatas must of type \
                            anndata.AnnData')
        
        # Ensuring all train/test indices are the same
        if check_uns:
            assert np.array_equal(adatas[0].uns['train_indices'],
                                    adata.uns['train_indices']), \
                "Train indices across AnnData objects in adatas do not \
                    match, ensure adata.uns['train_indices'] are the same \
                    for all elements"
                
            assert np.array_equal(adatas[0].uns['test_indices'], 
                                    adata.uns['test_indices']), \
                "Test indices across AnnData objects in adatas do not match, \
                    ensure adata.uns['test_indices'] are the same for all \
                    elements"

        # Ensuring all cell labels are the same 
        if check_obs:
            assert np.array_equal(adatas[0].obs['labels'], 
                                    adata.obs['labels']), \
                f"adata.obs['labels'] are different between AnnData objects"
    
    return None


def _eval_labels(cell_labels : np.ndarray, train_indices : np.ndarray, 
                  test_indices : np.ndarray, 
                  remove_labels : bool = False) -> np.ndarray:
    '''
    Takes an array of multiclass cell labels and returns a unique array 
    of cell labels to test for.
    Args:
        cell_labels - a numpy array of cell labels that coorespond to 
            an AnnData object.
        train_indices - a numpy array of indices for the training 
            samples in an AnnData object.
        test_indices - a numpy array of indices for the testing samples 
            in an AnnData object.
        remove_labels - If True, models will only be created for cell 
            labels in both the training and test data, if False, 
            models will be generated for all cell labels in the 
            training data.
    Returns:
        Returns a numpy array of unique cell labels to be iterated 
        through during one versus all experimental setups.
    '''
    train_uniq_labels = np.unique(cell_labels[train_indices])
    test_uniq_labels = np.unique(cell_labels[test_indices])

    # Ensuring that at least one cell type label between the two data
    #   are the same
    assert len(np.intersect1d(train_uniq_labels, test_uniq_labels)) > 0, \
        "There are no common labels between cells in the training and \
            testing samples"

    if np.array_equal(train_uniq_labels, test_uniq_labels):
        uniq_labels = train_uniq_labels
    else:
        # cells in the training but not the testing
        not_in_test = np.setdiff1d(train_uniq_labels, test_uniq_labels)
        # cells in the testing but not the training
        not_in_train = np.setdiff1d(test_uniq_labels, train_uniq_labels)    

        if len(not_in_test) > 0:
            print(f'WARNING: {not_in_test} cells are not in test data',
                flush = True)

        if len(not_in_train) > 0:
            print(f'WARNING: {not_in_train} cells are not in train data \
                  and will not be tested for',
                  flush = True)

        if remove_labels:
            uniq_labels = np.intersect1d(train_uniq_labels, test_uniq_labels)
        else:
            uniq_labels = train_uniq_labels

    return uniq_labels


def run(adata : ad.AnnData, alpha_list : np.ndarray, 
        metrics : list | None = None) -> dict:
    '''
    Takes a processed AnnData object and creates a model for each
    element in alpha_list. Return a dictionary with metrics and 
    predictions.
    Args:
        adata - an AnnData object with 'Z_train', 'Z_test', and 
            'group_dict' keys in adata.uns.
        alpha_list - a np.ndarray of alpha values to create models 
            using. Alpha refers to the penalty parameter in Group
            Lasso. Larger alphas force group weights to shrink towards
            0 while smaller alphas apply a lesser penalty to kernal 
            weights.
        metrics - a list of strings where each element is a type of 
            metric to be calculated. Options are ['AUROC', 'F1-Score',
            'Accuracy', 'Precision', 'Recall']. When set to None, all
            metrics are calculated.
    Returns:
        A dictionary with keys and values: 
            'Metrics' : a nested dictionary as [alpha][metric] = value
            'Selected_groups' : a dictionary as [alpha] = array of 
                groups with nonzero weights
            'Norms' : a dictionary as [alpha] = array of kernel weights
                for each group, order respective to 'Group_names'
            'Predictions' : a dictionary as [alpha] = predicted class
                respective to 'Observations' for that alpha
            'Observed' : an array of ground truth cell labels from the
                test set 
            'Test_indices' : indices of samples respective to adata 
                used in the training set
            'Group_names' : an array of group names respective to each
                array in 'Norms'
            'Model' : a dictionary where [alpha] = Celer Group Lasso
                object for that alpha
            'RAM_usage' : memory usage after training models for each 
                alpha
    '''
    if metrics == None:
        metrics = ['AUROC', 'F1-Score','Accuracy', 'Precision', 'Recall']

    # Initializing variables to capture metrics
    group_names = list(adata.uns['group_dict'].keys())
    predicted = {}
    group_norms = {}
    metric_dict = {}
    selected_pathways = {}
    train_time = {}
    models = {}

    D = adata.uns['D']

    # Generating models for each alpha and outputs
    for alpha in alpha_list:
        
        print(f'  Evaluating model. Alpha: {alpha}', flush = True)

        train_start = time.time()

        adata = train_model(adata, group_size= 2*D, alpha = alpha)
        predicted[alpha], metric_dict[alpha] = predict(adata, 
                                                            metrics = metrics)
        selected_pathways[alpha] = find_selected_groups(adata)

        kernel_weights = adata.uns['model'].coef_
        group_norms[alpha] = [
            np.linalg.norm(kernel_weights[i * 2 * D : (i + 1) * 2 * D - 1])
            for i in np.arange(len(group_names))
            ]
        
        models[alpha] = adata.uns['model']
        
        train_end = time.time()
        train_time[alpha] = train_end - train_start

    # Combining results into one object
    results = {}
    results['Metrics'] = metric_dict
    results['Selected_pathways'] = selected_pathways
    results['Norms'] = group_norms
    results['Predictions'] = predicted
    results['Observed'] = adata.obs['labels'].iloc[adata.uns['test_indices']]
    results['Test_indices'] = adata.uns['test_indices']
    results['Group_names']= group_names
    results['Models'] = models
    results['Train_time'] = train_time
    results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'

    return results



def one_v_all(adatas : list, names : list, alpha_list : np.ndarray, 
              tfidf : list, D : int, remove_labels = False) -> dict:
    '''
    This function takes a list of AnnData objects and creates a model
    for each cell label type.
    Args:
        adatas - a list of AnnData objects created by create_adata()
            where each AnnData is one modality and composed of both 
            training and testing samples. Requires that train_indices
            and test_indices are the same across all AnnDatas.
        names - a list of string variables that describe each modality
            respective to adatas for labeling.
        cell_labels - as scMKL is a binary classifier, there can only
            be two cell label types in the adata.obs['labels'] slot.
            Given this, np.ndarray is required to generate custom cell
            labels for each type present in the data. Array should 
            coorespond to samples in AnnData objects.
        remove_labels - If True, models will only be created for cell
            labels in both the training and test data, if False, models
            will be generated for all cell labels in the training data.
    Returns:
        Returns a dictionary of metrics, group weights, trained models
        for each label vs. the rest, predictions for each model, and a
        consensus prediction array for each sample in the train set.
    '''
    # Formatting checks ensuring all adata elements are 
    # AnnData objects and train/test indices are all the same
    _check_adatas(adatas, check_obs = True, check_uns = True)

    # Extracting train and test indices
    train_indices = adatas[0].uns['train_indices']
    test_indices = adatas[0].uns['test_indices']

    # Checking and capturing cell labels
    uniq_labels = _eval_labels(  cell_labels = adatas[0].obs['labels'], 
                                train_indices = train_indices,
                                 test_indices = test_indices,
                                remove_labels = remove_labels)


    # Calculating Z matrices, method depends on whether there are multiple 
    # adatas (modalities)
    if len(adatas) == 1:
        adata = estimate_sigma(adatas[0], n_features = 200)
        adata = calculate_z(adata, n_features = 5000)
    else:
        adata = multimodal_processing(adatas = adatas, 
                                            names = names, 
                                            tfidf = tfidf, 
                                            z_calculation = True)

    del adatas
    gc.collect()

    # Initializing for capturing model outputs
    results = {}

    # Capturing cell labels before overwriting
    cell_labels = np.array(adata.obs['labels'])

    for label in uniq_labels:
        print(f"Comparing {label} to other types", flush = True)
        cur_labels = cell_labels.copy()
        cur_labels[cell_labels != label] = 'other'
        
        # Replacing cell labels for current cell type vs rest
        adata.obs['labels'] = cur_labels

        # Running scMKL
        results[label] = run(adata, alpha_list)


    return results