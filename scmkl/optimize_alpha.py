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


def get_labels(adata: list | ad.AnnData):

    train_indices = adata[0].uns['train_indices'].copy()
    y = adata[0].obs['labels'].iloc[train_indices].to_numpy()

    return train_indices, y


def get_folds(y: np.array, k: int):
    """
    
    """
    # Splits the labels evenly between folds
    pos_idcs = np.where(y == np.unique(y)[0])[0]
    neg_idcs = np.setdiff1d(np.arange(len(y)), pos_idcs)
    
    pos_anno = np.arange(len(pos_idcs)) % k
    neg_anno = np.arange(len(neg_idcs)) % k

    return pos_idcs, neg_idcs, pos_anno, neg_anno


def bin_optimize_alpha(adata: ad.AnnData | list[ad.AnnData], 
                       group_size: int | None=None, 
                       tfidf: list[bool]=False, 
                       alpha_array: np.ndarray=default_alphas, 
                       k: int=4, metric: str='AUROC', 
                       early_stopping: bool=False,
                       batches: int=10, batch_size: int=100,
                       combination: str='concatenate'):
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
        
    train_indices, y = get_labels(adata)

    pos_idcs, neg_idcs, pos_anno, neg_anno = get_folds(y, k)

    metric_array = np.zeros((len(alpha_array), k))

    gc.collect()

    for fold in np.arange(k):
        cv_adata = [adata[i][train_indices, :] 
                    for i in range(len(adata))]
        
        for i in range(len(cv_adata)):
            if 'sigma' in cv_adata[i].uns_keys():
                del cv_adata[i].uns['sigma']

        # Create CV train/test indices
        fold_train = np.concatenate((pos_idcs[np.where(pos_anno != fold)[0]], 
                                     neg_idcs[np.where(neg_anno != fold)[0]]))
        fold_test = np.concatenate((pos_idcs[np.where(pos_anno == fold)[0]], 
                                    neg_idcs[np.where(neg_anno == fold)[0]]))


        for i in range(len(cv_adata)):
            cv_adata[i].uns['train_indices'] = fold_train
            cv_adata[i].uns['test_indices'] = fold_test
            if tfidf[i]:
                cv_adata[i] = tfidf_normalize(cv_adata[i], binarize=True)

            cv_adata[i] = calculate_z(cv_adata[i], n_features= 5000, 
                            batches = batches, batch_size = batch_size)
            
        names = ['Adata ' + str(i + 1) for i in range(len(cv_adata))]
        cv_adata = multimodal_processing(adata, names, tfidf, 
                                            combination, batches, 
                                            batch_size)
                    
            
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

        del cv_adata
        gc.collect()

    # Take AUROC mean across the k folds to find alpha yielding highest AUROC
    alpha_star = alpha_array[np.argmax(np.mean(metric_array, axis = 1))]
    gc.collect()

    return alpha_star


def multiclass_optimize_alpha(adata: ad.AnnData | list[ad.AnnData], 
                   group_size: int, 
                   tfidf: list[bool]=[False], 
                   alpha_array: np.ndarray=default_alphas, 
                   k: int=4, metric: str='AUROC', early_stopping: bool=False,
                   batches: int=10, batch_size: int=100, 
                   force_balance: bool=True, combination: str='concatenate',
                   train_dict: dict=None):
    """
    
    """
    classes = np.unique(adata[0].obs['labels'])
    orig_labels = adata[0].obs['labels'].to_numpy().copy()
    orig_train = adata[0].uns['train_indices'].copy()

    if train_dict:
        train_idcs = train_dict
    else:
        if force_balance:
            train_idcs = get_class_train(adata[0].uns['train_indices'], 
                                         adata[0].obs['labels'], 
                                         adata[0].uns['seed_obj'])
        else:
            train_idcs = {ct: adata[0].uns['train_indices'].copy()
                          for ct in classes}

    opt_alpha_dict = dict()

    for cl in classes:
        temp_classes = orig_labels.copy()
        temp_classes[temp_classes != cl] = 'other'

        for i in range(len(adata)):
            adata[i].obs['labels'] = temp_classes.copy()
            adata[i].uns['train_indices'] = train_idcs[cl]

        opt_alpha_dict[cl] = bin_optimize_alpha(adata, group_size, tfidf, 
                                                alpha_array, k, metric, 
                                                early_stopping, batches, 
                                                batch_size, combination)     
    
    # Global adata obj will be permanently changed if not reset
    for i in range(len(adata)):
        adata[i].obs['labels'] = orig_labels
        adata[i].uns['train_indices'] = orig_train

    return opt_alpha_dict


def optimize_alpha(adata: ad.AnnData | list[ad.AnnData], 
                   group_size: int | None=None, 
                   tfidf: None | list[bool]=None, 
                   alpha_array: np.ndarray=default_alphas, 
                   k: int=4, metric: str='AUROC', early_stopping: bool=False,
                   batches: int=10, batch_size: int=100, 
                   combination: str='concatenate', force_balance: bool=True,
                   train_dict: dict=None):
    """
    
    """
    # Need singe-view runs to be iterable
    if isinstance(adata, ad.AnnData):
        adata = [adata.copy()]

    if isinstance(tfidf, type(None)):
        tfidf = len(adata)*[False]

    is_multi = 2 < len(set(adata[0].obs['labels']))
    
    if isinstance(group_size, type(None)):
        group_size = 2*adata[0].uns['D']


    if is_multi:
        alpha_star = multiclass_optimize_alpha(adata, group_size, tfidf, 
                                                alpha_array, k, metric, 
                                                early_stopping, batches, 
                                                batch_size, force_balance, 
                                                combination, train_dict)
        
    else:
        alpha_star = bin_optimize_alpha(adata, group_size, tfidf, 
                                        alpha_array, k, metric, 
                                        early_stopping, batches, 
                                        batch_size, combination)
        
    return alpha_star
