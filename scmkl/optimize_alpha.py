import numpy as np
import anndata as ad
import gc
import tracemalloc

from scmkl.tfidf_normalize import tfidf_normalize
from scmkl.calculate_z import calculate_z
from scmkl.train_model import train_model
from scmkl.multimodal_processing import multimodal_processing
from scmkl.test import predict
from scmkl.one_v_rest import get_class_train


# Array of alphas to be used if not provided
default_alphas = np.round(np.linspace(1.9, 0.05, 10),2)


def stop_early(metric_array, alpha_idx, fold_idx):
    """
    Assumes smallest alpha comes first.
    """
    # Must be at least two metrics from two alphas to compare
    if alpha_idx <= 0:
        return False
    
    cur_met = metric_array[alpha_idx, fold_idx]
    last_met = metric_array[alpha_idx - 1, fold_idx]

    if cur_met < last_met:
        return True
    else:
        return False


def sort_alphas(alpha_array: np.ndarray):
    """
    Sorts alphas from smallest to largest.
    """
    order = np.argsort(alpha_array)
    alpha_array = alpha_array[order]

    return alpha_array


def multimodal_optimize_alpha(adatas: list[ad.AnnData], group_size: int, 
                              tfidf_list: list | bool=False,
                              alpha_array: np.ndarray=default_alphas, 
                              k: int=4, metric: str='AUROC', 
                              early_stopping: bool=False,
                              batches: int=10, batch_size: int=100):
    """
    binary multimodal optimize alpha
    Iteratively train a grouplasso model and update alpha to find the 
    parameter yielding the desired sparsity. Meant to find a good 
    starting point for your model, and the alpha may need further fine 
    tuning.
    
    Parameters
    ----------
    adatas : list[ad.AnnData]
        Objects of type `ad.AnnData` where each object is one modality 
        and Z_train and Z_test are calculated

    group_size : None | int
        Argument describing how the features are grouped. If `None`, 
        `2 * adata.uns['D']` will be used. For more information see 
        [celer documentation](https://mathurinm.github.io/celer/
        generated/celer.GroupLasso.html).

    tfidf_list : list | None
        A boolean mask where `tfidf_list[i]` is respective to 
        `adatas[i]`. If `True`, TF-IDF normalization will be applied to 
        the respective `ad.AnnData` during cross validation
    
    alpha_array : np.ndarray
        All alpha values to be tested.

    k : int
        Number of folds to perform cross validation over.

    metric : str
        Which metric to use to optimize alpha. Options are `'AUROC'`, 
        `'Accuracy'`, `'F1-Score'`, `'Precision'`, and `'Recall'`.

    batches : int
        The number of batches to use for the distance calculation.
        This will average the result of `batches` distance calculations
        of `batch_size` randomly sampled cells. More batches will converge
        to population distance values at the cost of scalability.

    batch_size : int
        The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability. If 
        `batches*batch_size > num_training_cells`, `batch_size` will be 
        reduced to `int(num_training_cells/batches)`.
            
    Returns
    -------
    alpha_star : float
        The alpha value yielding the best performing model from cross 
        validation.
    """
    assert isinstance(k, int) and k > 0, 'Must be a positive integer number of folds'

    import warnings 
    warnings.filterwarnings('ignore')

    # Sorting alphas smallest to largers
    alpha_array = sort_alphas(alpha_array)

    if not tfidf_list:
        tfidf_list = [False]*len(adatas)

    y = adatas[0].obs['labels'].iloc[adatas[0].uns['train_indices']].to_numpy()
    
    # Splits the labels evenly between folds
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)

    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k

    metric_array = np.zeros((len(alpha_array), k))

    cv_adatas = []

    for adata in adatas:
        cv_adatas.append(adata[adata.uns['train_indices'],:].copy())

    del adatas
    gc.collect()

    for fold in np.arange(k):

        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], 
                                     negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], 
                                    negative_indices[np.where(negative_annotations == fold)[0]]))

        for i in range(len(cv_adatas)):
            cv_adatas[i].uns['train_indices'] = fold_train
            cv_adatas[i].uns['test_indices'] = fold_test

        # Creating dummy names for cv. 
        # Necessary for interpretability but not for AUROC cv
        dummy_names = [f'adata {i}' for i in range(len(cv_adatas))]

        # Calculate the Z's for each modality independently
        fold_cv_adata = multimodal_processing(adatas = cv_adatas, 
                                              names = dummy_names, 
                                              tfidf = tfidf_list, 
                                              batch_size= batch_size, 
                                              batches = batches)

        fold_cv_adata.uns['seed_obj'] = cv_adatas[0].uns['seed_obj']

        if 'sigma' in fold_cv_adata.uns_keys():
            del fold_cv_adata.uns['sigma']

        # In train_model we index Z_train for balancing multiclass labels. We just recreate
        # dummy indices here that are unused for use in the binary case
        fold_cv_adata.uns['train_indices'] = np.arange(0, len(fold_train))

        gc.collect()

        for j, alpha in enumerate(alpha_array):

            fold_cv_adata = train_model(fold_cv_adata, group_size, alpha = alpha)

            _, metrics = predict(fold_cv_adata, metrics = [metric])
            metric_array[j, fold] = metrics[metric]

            # If metrics are decreasing, cv stopped and moving to next fold
            end_fold = stop_early(metric_array, alpha_idx=j, fold_idx=fold)
            if end_fold and early_stopping:
                break

        del fold_cv_adata
        gc.collect()

    # Take AUROC mean across the k folds and select the alpha resulting in highest AUROC
    alpha_star = alpha_array[np.argmax(np.mean(metric_array, axis = 1))]
    del cv_adatas
    gc.collect()
    
    return alpha_star


def bin_optimize_alpha(adata: ad.AnnData | list[ad.AnnData], 
                       group_size: int | None=None, 
                       tfidf: bool | list[bool]=False, 
                       alpha_array: np.ndarray=default_alphas, 
                       k: int=4, metric: str='AUROC', 
                       early_stopping: bool=False,
                       batches: int=10, batch_size: int=100):
    """
    binary optimize_alpha
    Iteratively train a grouplasso model and update alpha to find the 
    parameter yielding best performing sparsity. This function 
    currently only works for binary experiments.

    Parameters
    ----------
    adata : ad.AnnData | list[ad.AnnData]
        `ad.AnnData`(s) with `'Z_train'` and `'Z_test'` in 
        `adata.uns.keys()`.

    group_size : None | int
        Argument describing how the features are grouped. If `None`, 
        `2 * adata.uns['D']` will be used. For more information see 
        [celer documentation](https://mathurinm.github.io/celer/
        generated/celer.GroupLasso.html).

    tfidf : list | bool
        If `False`, no data will be TF-IDF transformed. If 
        `type(adata) is list` and TF-IDF transformation is desired for 
        all or some of the data, a bool list corresponding to `adata` 
        must be provided. To simply TF-IDF transform `adata` when 
        `type(adata) is ad.AnnData`, use `True`.
    
    alpha_array : np.ndarray
        Array of all alpha values to be tested.

    k : int
        Number of folds to perform cross validation over.
            
    metric : str
        Which metric to use to optimize alpha. Options are `'AUROC'`, 
        `'Accuracy'`, `'F1-Score'`, `'Precision'`, and `'Recall'`.

    batches : int
        The number of batches to use for the distance calculation.
        This will average the result of `batches` distance calculations
        of `batch_size` randomly sampled cells. More batches will converge
        to population distance values at the cost of scalability.

    batch_size : int
        The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability. If 
        `batches*batch_size > num_training_cells`, `batch_size` will be 
        reduced to `int(num_training_cells/batches)`.

    Returns
    -------
    alpha_star : float
        The best performing alpha value from cross validation on 
        training data.

    Examples
    --------
    >>> alpha_star = scmkl.optimize_alpha(adata)
    >>> alpha_star
    0.1
    """    
    assert isinstance(k, int) and k > 0, "'k' must be positive"

    import warnings 
    warnings.filterwarnings('ignore')

    # Sorting alphas smallest to largers
    alpha_array = sort_alphas(alpha_array)

    if group_size == None:
        group_size = adata.uns['D']*2

    if type(adata) == list:
        alpha_star = multimodal_optimize_alpha(adatas = adata, 
                                               group_size = group_size,
                                               tfidf_list = tfidf, 
                                               alpha_array = alpha_array, 
                                               metric = metric,
                                               batch_size = batch_size,
                                               batches = batches)
        return alpha_star

    y = adata.obs['labels'].iloc[adata.uns['train_indices']].to_numpy()
    
    # Splits the labels evenly between folds
    positive_indices = np.where(y == np.unique(y)[0])[0]
    negative_indices = np.setdiff1d(np.arange(len(y)), positive_indices)
    
    positive_annotations = np.arange(len(positive_indices)) % k
    negative_annotations = np.arange(len(negative_indices)) % k

    metric_array = np.zeros((len(alpha_array), k))

    gc.collect()

    for fold in np.arange(k):
        cv_adata = adata[adata.uns['train_indices'],:]

        if 'sigma' in cv_adata.uns_keys():
            del cv_adata.uns['sigma']

        # Create CV train/test indices
        fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], 
                                     negative_indices[np.where(negative_annotations != fold)[0]]))
        fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], 
                                    negative_indices[np.where(negative_annotations == fold)[0]]))

        cv_adata.uns['train_indices'] = fold_train
        cv_adata.uns['test_indices'] = fold_test

        if tfidf:
            cv_adata = tfidf_normalize(cv_adata, binarize= True)

        # Estimating kernel widths and calculating Zs
        cv_adata = calculate_z(cv_adata, n_features= 5000, 
                               batches = batches, batch_size = batch_size)

        # In train_model we index Z_train for balancing multiclass labels. We just recreate
        # dummy indices here that are unused for use in the binary case
        cv_adata.uns['train_indices'] = np.arange(0, len(fold_train))

        gc.collect()

        for i, alpha in enumerate(alpha_array):

            cv_adata = train_model(cv_adata, group_size, alpha = alpha)
            _, metrics = predict(cv_adata, metrics = [metric])
            metric_array[i, fold] = metrics[metric]

            # If metrics are decreasing, cv stopped and moving to next fold
            end_fold = stop_early(metric_array, alpha_idx=i, fold_idx=fold)
            if end_fold and early_stopping:
                break

            gc.collect()

        del cv_adata
        gc.collect()

    # Take AUROC mean across the k folds to find alpha yielding highest AUROC
    alpha_star = alpha_array[np.argmax(np.mean(metric_array, axis = 1))]
    gc.collect()

    return alpha_star


def multiclass_optimize_alpha(adata: ad.AnnData | list[ad.AnnData], 
                   group_size: int | None=None, 
                   tfidf: bool | list[bool]=False, 
                   alpha_array: np.ndarray=default_alphas, 
                   k: int=4, metric: str='AUROC', early_stopping: bool=False,
                   batches: int=10, batch_size: int=100):
    """
    
    """
    if isinstance(adata, ad.AnnData):
        classes = np.unique(adata.obs['labels'])
        orig_labels = adata.obs['labels'].to_numpy().copy()
        orig_train = adata.uns['train_indices'].copy()
        balanced_idcs = get_class_train(adata.uns['train_indices'], 
                                    adata.obs['labels'], 
                                    adata.uns['seed_obj'])
    else:
        classes = np.unique(adata[0].obs['labels'])
        orig_labels = adata[0].obs['labels'].to_numpy().copy()
        orig_train = adata[0].uns['train_indices'].copy()
        balanced_idcs = get_class_train(adata[0].uns['train_indices'], 
                                    adata[0].obs['labels'], 
                                    adata[0].uns['seed_obj'])

    opt_alpha_dict = dict()

    for cl in classes:
        temp_classes = orig_labels.copy()
        temp_classes[temp_classes != cl] = 'other'

        # Adding binarized labels and balanced test indices to adata(s)
        if isinstance(adata, ad.AnnData):
            adata.obs['labels'] = temp_classes.copy()
            adata.uns['train_indices'] = balanced_idcs[cl]
            
            opt_alpha_dict[cl] = bin_optimize_alpha(adata, 
                                                    group_size, 
                                                    tfidf, 
                                                    alpha_array, 
                                                    k, 
                                                    metric, 
                                                    early_stopping,
                                                    batches, 
                                                    batch_size)     
        else: 
            for i in range(len(adata)):
                adata[i].obs['labels'] = temp_classes.copy()
                adata[i].uns['train_indices'] = balanced_idcs[cl]

            opt_alpha_dict[cl] = multimodal_optimize_alpha(adata, 
                                                            group_size, 
                                                            tfidf, 
                                                            alpha_array, 
                                                            k, 
                                                            metric, 
                                                            early_stopping, 
                                                            batches, 
                                                            batch_size)
        
        
        
    # Global adata obj will be permanently changed if not reset
    if isinstance(adata, ad.AnnData):
            adata.obs['labels'] = orig_labels
            adata.uns['train_indices'] = orig_train
            
    else: 
        for i in range(len(adata)):
            adata[i].obs['labels'] = orig_labels
            adata[i].uns['train_indices'] = orig_train
        
    return opt_alpha_dict


def optimize_alpha(adata: ad.AnnData | list[ad.AnnData], 
                   group_size: int | None=None, 
                   tfidf: bool | list[bool]=False, 
                   alpha_array: np.ndarray=default_alphas, 
                   k: int=4, metric: str='AUROC', early_stopping: bool=False,
                   batches: int=10, batch_size: int=100):
    """
    
    """
    is_adata = isinstance(adata, ad.AnnData)
    
    if is_adata:
        is_multi = len(set(adata.obs['labels'])) > 2
    else:
        is_multi = len(set(adata[0].obs['labels'])) > 2

    if is_multi:
        alpha_star = multiclass_optimize_alpha(adata, group_size, tfidf, 
                                                alpha_array, k, metric, 
                                                early_stopping, batches, 
                                                batch_size)
        
    else:
        alpha_star = bin_optimize_alpha(adata, group_size, tfidf, 
                                        alpha_array, k, metric, 
                                        early_stopping, batches, 
                                        batch_size)
        
    return alpha_star
