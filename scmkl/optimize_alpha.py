import numpy as np
import anndata as ad
import gc
from sklearn.model_selection import StratifiedKFold

from scmkl.tfidf_normalize import tfidf_normalize
from scmkl.calculate_z import calculate_z
from scmkl.train_model import train_model
from scmkl.multimodal_processing import multimodal_processing
from scmkl.test import predict
from scmkl.one_v_rest import get_class_train
from scmkl.create_adata import sort_samples


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


def get_folds(adata: ad.AnnData, k: int):
    """
    With labels of samples for cross validation and number of folds, 
    returns the indices and label for each k-folds.

    Parameters
    ----------
    adata : ad.AnnData
        `AnnData` object containing `'labels'` column in `.obs` and 
        `'train_indices'` in `.uns`.

    k : int
        The number of folds to perform cross validation over.

    Returns
    -------
    folds : dict
        A dictionary with keys being [0, k) and values being a tuple 
        with first element being training sample indices and the second 
        being testing value indices.

    Examples
    --------
    >>> adata = scmkl.create_adata(...)
    >>> folds = scmkl.get_folds(adata, k=4)
    >>>
    >>> train_fold_0, test_fold_0 = folds[0]
    >>>
    >>> train_fold_0
    array([  0,   3,   5,   6,   8,   9,  10,  11,  12,  13])
    >>> test_fold_0
    array([  1,   2,  4,  7])
    """
    y = adata.obs['labels'][adata.uns['train_indices']].copy()

    # Creating dummy x prevents holding more views in memory
    x = y.copy()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=100)

    folds = dict()
    for fold, (fold_train, fold_test) in enumerate(skf.split(x, y)):
        folds[fold] = (fold_train, fold_test)

    return folds


def prepare_fold(fold_adata, sort_idcs, fold_train, fold_test):
    """
    Reorders adata samples, reassigns each `adata.uns['train_indices' & 
    'test_indices']`, and removes sigmas if present.

    Parameters
    ----------
    fold_adata : list[ad.AnnData]
        A `list` of `AnnData` objects to be reordered based on fold 
        indices.

    sort_idcs: np.ndarray
        The indices that will sort `AnnData`s as all train samples then 
        all test.

    fold_train : np.ndarray
        The indices of training sample for respective fold.

    fold_test : np.ndarray
        The indices of testing samples for respective fold.

    Returns
    -------
    fold_adata : list[ad.AnnData]
        The `list` of `AnnData`s with each `AnnData` sorted as all 
        training samples then all testing samples. 
        `adata.uns['train_indices' & 'test_indices']` are also updated 
        to reflect new sample positions.

    Examples
    --------
    >>> adata = [scmkl.create_adata(...)]
    >>> folds = scmkl.get_folds(adata[0], k=4)
    >>> sort_idcs, fold_train, fold_test = sort_samples(fold_train, 
    ...                                                 fold_test)
    >>> fold_adata = scmkl.prepare_fold(adata, sort_idcs, 
    ...                                 fold_train, fold_test)
    """
    for i in range(len(fold_adata)):
        fold_adata[i] = fold_adata[i][sort_idcs]
        fold_adata[i].uns['train_indices'] = fold_train
        fold_adata[i].uns['test_indices'] = fold_test
    
    # Need to recalculate sigmas for new fold train
    for i in range(len(fold_adata)):
        if 'sigma' in fold_adata[i].uns_keys():
            del fold_adata[i].uns['sigma']

    return fold_adata


def process_fold(fold_adata, names, tfidf, combination, batches, batch_size):
    """
    Combines adata if needed, estimates sigmas, and calculates kernels 
    for model training and evaluation.

    Parameters
    ----------
    fold_adata : list[ad.AnnData]
        A `list` of `AnnData` objects to combine and calculate kernels 
        for.

    names : list
        A list of names respective to each `AnnData` in `fold_adata` 
        for verbose outputs.

    tfidf : list[bool]
        A boolean list indicating whether or not to perform TF-IDF 
        transformation for each adata respective to `fold_adata`.

    combination : str
        The method of combining `AnnData` objects passed to 
        `ad.concatenate()`. Ignored if `len(fold_adata) == 1`.

    batches : int
        The number of batches for kernel width (sigma) estimation.

    batch_size : int
        The number of samples to include in each batch of kernel width 
        (sigma) estimation.

    Examples
    --------
    >>> adata = [scmkl.create_adata(...)]
    >>> tfidf = [False]
    >>> names = ['adata1']
    >>> folds = scmkl.get_folds(adata[0], k=4)
    >>> sort_idcs, fold_train, fold_test = sort_samples(fold_train, 
    ...                                                 fold_test)
    >>> fold_adata = scmkl.prepare_fold(adata, sort_idcs, 
    ...                                 fold_train, fold_test)
    >>> fold_adata = scmkl.process_fold(fold_adata, names, tfidf)
    """
    if 1 < len(fold_adata):
        fold_adata = multimodal_processing(fold_adata, names, tfidf, 
                                           combination, batches, 
                                           batch_size, False)
    else:
        fold_adata = fold_adata[0]
        if tfidf[0]:
            fold_adata = tfidf_normalize(fold_adata, binarize=True)
        fold_adata = calculate_z(fold_adata, n_features= 5000, 
                                 batches=batches, batch_size=batch_size)
        
    return fold_adata


def bin_optimize_alpha(adata: list[ad.AnnData], 
                       group_size: int | None=None, 
                       tfidf: list[bool]=False, 
                       alpha_array: np.ndarray=default_alphas, 
                       k: int=4, metric: str='AUROC', 
                       early_stopping: bool=False,
                       batches: int=10, batch_size: int=100,
                       combination: str='concatenate'):
    """
    Iteratively train a grouplasso model and update alpha to find the 
    parameter yielding best performing sparsity via k-fold cross 
    validation. This function currently only works for binary 
    experiments. Called by `scmkl.optimize_alpha()`.

    Parameters
    ----------
    adata : list[ad.AnnData]
        List of `ad.AnnData`(s) with `'Z_train'` and `'Z_test'` in 
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

    alpha_array = sort_alphas(alpha_array)

    # Only want folds for training samples
    train_indices = adata[0].uns['train_indices'].copy()
    cv_adata = [adata[i][train_indices, :].copy()
                for i in range(len(adata))]

    folds = get_folds(adata[0], k)

    metric_array = np.zeros((len(alpha_array), k))

    for fold in range(k):

        fold_train, fold_test = folds[fold]
        fold_adata = cv_adata.copy()

        # Downstream functions expect train then test samples in adata(s)
        sort_idcs, fold_train, fold_test = sort_samples(fold_train, fold_test)
        fold_adata = prepare_fold(fold_adata, sort_idcs, 
                                  fold_train, fold_test)
            
        names = ['Adata ' + str(i + 1) for i in range(len(cv_adata))]

        # Adatas need combined if applicable and kernels calculated 
        fold_adata = process_fold(fold_adata, names, tfidf, combination, 
                                  batches, batch_size)

        for i, alpha in enumerate(alpha_array):

            fold_adata = train_model(fold_adata, group_size, alpha=alpha)
            _, metrics = predict(fold_adata, metrics=[metric])
            metric_array[i, fold] = metrics[metric]

            # If metrics are decreasing, cv stopped and moving to next fold
            end_fold = stop_early(metric_array, alpha_idx=i, fold_idx=fold)
            if end_fold and early_stopping:
                break

        del fold_adata
        gc.collect()

    del cv_adata
    gc.collect()

    # Need highest performing alpha for given metric
    alpha_star = alpha_array[np.argmax(np.mean(metric_array, axis = 1))]

    return alpha_star


def multi_optimize_alpha(adata: list[ad.AnnData], group_size: int, 
                         tfidf: list[bool]=[False], 
                         alpha_array: np.ndarray=default_alphas, k: int=4, 
                         metric: str='AUROC', early_stopping: bool=False,
                         batches: int=10, batch_size: int=100, 
                         force_balance: bool=True, combination: str='concatenate',
                         train_dict: dict=None):
    """
    Wrapper function for running k-fold cross validation for every 
    label in a multiclass experiment. Called by 
    `scmkl.optimize_alpha()`. 

    Parameters
    ----------
    adata : list[ad.AnnData]
        List of `ad.AnnData`(s) with `'Z_train'` and `'Z_test'` in 
        `adata.uns.keys()`.

    group_size : None | int
        Argument describing how the features are grouped. If `None`, 
        `2 * adata.uns['D']` will be used. For more information see 
        [celer documentation](https://mathurinm.github.io/celer/
        generated/celer.GroupLasso.html).

    tfidf : list[bool]
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

    force_balance: bool
        If `True`, training sets will be balanced to reduce class label 
        imbalance for each iteration. Defaults to `False`.

    other_factor : float
        The ratio of cells to sample for the other class for each 
        model. For example, if classifying B cells with 100 B cells in 
        training, if `other_factor=1`, 100 cells that are not B cells 
        will be trained on with the B cells. This will be done for each 
        fold for each class if `force_balance` is `True`. 

    combination: str
        How should multiple views of data be combined. For more details 
        see ad.concat.

    train_dict: dict
        A `dict` where each key is a class label and values are are the 
        indices to be trained with for that class for class balance. 
        All values must be present in each adata.uns['train_indices'].

    Returns
    -------
    alpha_star : dict
        A dictionary with keys being class labels and values being the 
        best performing alpha parameter for that class as a float.
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
                   group_size: int | None=None, tfidf: None | list[bool]=None, 
                   alpha_array: np.ndarray=default_alphas, k: int=4, 
                   metric: str='AUROC', early_stopping: bool=False,
                   batches: int=10, batch_size: int=100, 
                   combination: str='concatenate', force_balance: bool=True,
                   train_dict: dict=None):
    """
    K-fold cross validation for optimizing alpha hyperparameter using 
    training indices. 

    Parameters
    ----------
    adata : list[ad.AnnData]
        List of `ad.AnnData`(s) with `'Z_train'` and `'Z_test'` in 
        `adata.uns.keys()`.

    group_size : None | int
        Argument describing how the features are grouped. If `None`, 
        `2 * adata.uns['D']` will be used. For more information see 
        [celer documentation](https://mathurinm.github.io/celer/
        generated/celer.GroupLasso.html).

    tfidf : list[bool]
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

    force_balance: bool
        If `True`, training sets will be balanced to reduce class label 
        imbalance for each iteration. Defaults to `False`.

    other_factor : float
        The ratio of cells to sample for the other class for each 
        model. For example, if classifying B cells with 100 B cells in 
        training, if `other_factor=1`, 100 cells that are not B cells 
        will be trained on with the B cells. This will be done for each 
        fold for each class if `force_balance` is `True`. 

    combination: str
        How should multiple views of data be combined. For more details 
        see ad.concat.

    train_dict: dict
        A `dict` where each key is a class label and values are are the 
        indices to be trained with for that class for class balance. 
        All values must be present in each adata.uns['train_indices'].

    Returns
    -------
    alpha_star : float | dict
        If number of classes is more than 2, a dictionary with keys 
        being class labels and values being the best performing alpha 
        parameter for that class as a float. Else, a float for 
        comparing the two classes.
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
        alpha_star = multi_optimize_alpha(adata, group_size, tfidf, 
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
