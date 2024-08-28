import numpy as np
import gc
import tracemalloc

from scmkl.tfidf import tfidf_normalize
from scmkl.estimate_sigma import estimate_sigma
from scmkl.approx_kernels import calculate_z
from scmkl.train import train_model
from scmkl.test import calculate_auroc, find_selected_groups
from scmkl.multimodal_processing import multimodal_processing


def optimize_alpha(adata, group_size, tfidf = False, alpha_array = np.round(np.linspace(1.9,0.1, 10),2), k = 4):
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
        tfidf- Boolean value to determine if TFIDF normalization should be run at each fold. True means that it will be performed.
        starting_alpha- The alpha value to start the search at.
        alpha_array- Numpy array of all alpha values to be tested
        k- number of folds to perform cross validation over
            
    Output:
        sparsity_dict- Dictionary with tested alpha as keys and the number of selected pathways as the values
        alpha- The alpha value yielding the number of selected groups closest to the target.
    '''

    assert isinstance(k, int) and k > 0, 'Must be a positive integer number of folds'

    import warnings 
    warnings.filterwarnings('ignore')

    y = adata.obs['labels'].iloc[adata.uns['train_indices']].to_numpy()
    
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)

    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k

    auc_array = np.zeros((len(alpha_array), k))

    gc.collect()

    for fold in np.arange(k):
        
        print(f'Fold {fold + 1}:\n Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')

        cv_adata = adata[adata.uns['train_indices'],:]

        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))

        cv_adata.uns['train_indices'] = fold_train
        cv_adata.uns['test_indices'] = fold_test

        if tfidf:
            cv_adata = tfidf_normalize(cv_adata, binarize= True)

        cv_adata = estimate_sigma(cv_adata, n_features = 200)
        cv_adata = calculate_z(cv_adata, n_features= 5000)

        gc.collect()

        for i, alpha in enumerate(alpha_array):

            
            # print(f'  1. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            # print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            cv_adata = train_model(cv_adata, group_size, alpha = alpha)

            # print(f'  2. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            # print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')

            auc_array[i, fold] = calculate_auroc(cv_adata)

            # print(f'  3. Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')
            # print(f'   Adata size: {sys.getsizeof(cv_adata) / 1e9}')
            gc.collect()

        del cv_adata
        gc.collect()
    # Take AUROC mean across the k folds
    alpha_star = alpha_array[np.argmax(np.mean(auc_array, axis = 1))]
    gc.collect()
    

    return alpha_star


def optimize_sparsity(adata, group_size, starting_alpha = 1.9, increment = 0.2, target = 1, n_iter = 10):
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
        model = train_model(adata, group_size, alpha)
        num_selected = len(find_selected_groups(adata))

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


def multimodal_optimize_alpha(adata1, adata2, group_size = 1, tfidf_list = [False, False], alpha_array = np.round(np.linspace(1.9,0.1, 10),2), k = 4):
    '''
    Iteratively train a grouplasso model and update alpha to find the parameter yielding the desired sparsity.
    This function is meant to find a good starting point for your model, and the alpha may need further fine tuning.
    Input:
        adataX- Anndata objects with Z_train and Z_test calculated
        group_size- Argument describing how the features are grouped. 
            From Celer documentation:
            "groupsint | list of ints | list of lists of ints.
                Partition of features used in the penalty on w. 
                    If an int is passed, groups are contiguous blocks of features, of size groups. 
                    If a list of ints is passed, groups are assumed to be contiguous, group number g being of size groups[g]. 
                    If a list of lists of ints is passed, groups[g] contains the feature indices of the group number g."
            If 1, model will behave identically to Lasso Regression.
        tifidf_list- a boolean mask where tfidf_list[0] and tfidf_list[1] are respective to adata1 and adata2
            If True, tfidf normalization will be applied to the respective adata during cross validation
        starting_alpha- The alpha value to start the search at.
        alpha_array- Numpy array of all alpha values to be tested
        k- number of folds to perform cross validation over
            
    Output:
        sparsity_dict- Dictionary with tested alpha as keys and the number of selected pathways as the values
        alpha- The alpha value yielding the number of selected groups closest to the target.
    '''

    assert isinstance(k, int) and k > 0, 'Must be a positive integer number of folds'

    import warnings 
    warnings.filterwarnings('ignore')

    y = adata1.obs['labels'].iloc[adata1.uns['train_indices']].to_numpy()
    
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)

    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k

    auc_array = np.zeros((len(alpha_array), k))

    cv_adata1 = adata1[adata1.uns['train_indices'],:]
    cv_adata2 = adata2[adata2.uns['train_indices'],:]

    del adata1, adata2
    gc.collect()

    for fold in np.arange(k):
        
        print(f'Fold {fold + 1}:\n Memory Usage: {[mem / 1e9 for mem in tracemalloc.get_traced_memory()]} GB')

        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))

        cv_adata1.uns['train_indices'] = fold_train
        cv_adata1.uns['test_indices'] = fold_test
        cv_adata2.uns['train_indices'] = fold_train
        cv_adata2.uns['test_indices'] = fold_test

        cv_adata = multimodal_processing('assay1', 'assay2', cv_adata1, cv_adata2, tfidf_list, z_calculation = True)
        cv_adata.uns['seed_obj'] = cv_adata1.uns['seed_obj']

        gc.collect()



        for i, alpha in enumerate(alpha_array):

            cv_adata = train_model(cv_adata, group_size, alpha = alpha)

            auc_array[i, fold] = calculate_auroc(cv_adata)

            # gc.collect()
        del cv_adata
        gc.collect()
    # Take AUROC mean across the k folds
    alpha_star = alpha_array[np.argmax(np.mean(auc_array, axis = 1))]
    del cv_adata1, cv_adata2
    gc.collect()
    
    return alpha_star