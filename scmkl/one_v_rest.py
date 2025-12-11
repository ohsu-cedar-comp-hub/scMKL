import gc
import tracemalloc
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import f1_score


from scmkl.run import run
from scmkl.calculate_z import calculate_z
from scmkl.multimodal_processing import multimodal_processing
from scmkl._checks import _check_adatas


def _eval_labels(cell_labels: np.ndarray, train_indices: np.ndarray, 
                  test_indices: np.ndarray) -> np.ndarray:
    """
    Takes an array of multiclass cell labels and returns a unique array 
    of cell labels to test for.

    Parameters
    ----------
    cell_labels : np.ndarray
        Cell labels that coorespond to an AnnData object.

    train_indices : np.ndarray
        Indices for the training samples in an AnnData object.
    
    test_indices : np.ndarray
        Indices for the testing samples in an AnnData object.

    remove_labels : bool
        If `True`, models will only be created for cell labels in both 
        the training and test data, if `False`, models will be generated
        for all cell labels in the training data.

    Returns
    -------
    uniq_labels : np.ndarray
        Returns a numpy array of unique cell labels to be iterated 
        through during one versus all setups.
    """
    train_uniq_labels = np.unique(cell_labels[train_indices])
    test_uniq_labels = np.unique(cell_labels[test_indices])

    # Getting only labels in both training and testing sets
    uniq_labels = np.intersect1d(train_uniq_labels, test_uniq_labels)

    # Ensuring that at least one cell type label between the two data
    #   are the same
    cl_intersect = np.intersect1d(train_uniq_labels, test_uniq_labels)
    assert len(cl_intersect) > 0, ("There are no common labels between cells "
                                   "in the training and testing samples")

    return uniq_labels


def get_prob_table(results : dict, alpha: float | dict):
    """
    Takes a results dictionary with class and probabilities keys and 
    returns a table of probabilities for each class and the most 
    probable class for each cell.

    Parameters
    ----------
    results : dict
        A nested dictionary that contains a dictionary for each class 
        containing probabilities for each cell class.

    alpha : float | dict
        A float for which model probabilities should be evaluated 
        for.

    Returns
    -------
    prob_table : pd.DataFrame
        Each column is a cell class and the elements are the
        class probability outputs from the model.

    pred_class : list[str]
        The most probable cell classes respective to the training set 
        cells. 

    low_conf : list[bool]
        A bool list where `True`, sample max probability is less than 
        0.5.
    """
    if isinstance(alpha, float):
        prob_table = {class_ : results[class_]['Probabilities'][alpha][class_]
                    for class_ in results.keys()}
    else:
        prob_table = {class_ : list()
                      for class_ in alpha.keys()}
        for class_ in results.keys():
            cur_alpha = alpha[class_]
            prob_table[class_] = results[class_]['Probabilities'][cur_alpha][class_]

    prob_table = pd.DataFrame(prob_table)

    pred_class = []
    maxes = []

    for i, row in prob_table.iterrows():
        row_max = np.max(row)
        indices = np.where(row == row_max)
        prediction = prob_table.columns[indices]

        if len(prediction) > 1:
            prediction = " and ".join(prediction)
        else:
            prediction = prediction[0]

        pred_class.append(prediction)
        maxes.append(row_max)

    maxes = np.round(maxes, 0)
    low_conf = np.invert(np.array(maxes, dtype = np.bool_))

    return prob_table, pred_class, low_conf


def per_model_summary(results: dict, uniq_labels: np.ndarray | list | tuple, 
                      alpha: float) -> pd.DataFrame:
    """
    Takes the results dictionary from `scmkl.one_v_rest()` and adds a 
    summary dataframe show metrics for each model generated from the 
    runs.

    Parameters
    ----------
    results : dict
        Results from `scmkl.one_v_rest()`.

    uniq_labels : array_like
        Unique cell classes from the runs.

    alpha : float | dict
        The alpha for creating the summary from.

    Returns
    -------
    summary_df : pd.DataFrame
        Dataframe with classes on rows and metrics as cols.
    """
    # Getting metrics availible in results
    if isinstance(alpha, dict):
        alpha_key = list(alpha.keys())[0]
        alpha_key = alpha[alpha_key]
        avail_mets = list(results[uniq_labels[0]]['Metrics'][alpha_key])
    else:
        avail_mets = list(results[uniq_labels[0]]['Metrics'][alpha])

    summary_df = {metric : list()
                  for metric in avail_mets}
    summary_df['Class'] = uniq_labels

    for lab in summary_df['Class']:
        for met in avail_mets:
            if isinstance(alpha, dict):
                cur_alpha = alpha[lab]
            else:
                cur_alpha = alpha

            val = results[lab]['Metrics'][cur_alpha][met]
            summary_df[met].append(val)

    return pd.DataFrame(summary_df)


def get_class_train(train_indices: np.ndarray,
                    cell_labels: np.ndarray | list | pd.Series,
                    seed_obj: np.random._generator.Generator,
                    other_factor = 1.5):
    """
    This function returns a dict with each entry being a set of 
    training indices for each cell class to be used in 
    `scmkl.one_v_rest()`.

    Parameters
    ----------
    train_indices : np.ndarray
        The indices in the `ad.AnnData` object of samples availible to 
        train on.

    cell_labels : array_like
        The identity of all cells in the anndata object.

    seed_obj : np.random._generator.Generator
        The seed object used to randomly sample non-target samples.

    other_factor : float
        The ratio of cells to sample for the other class for each 
        model. For example, if classifying B cells with 100 B cells in 
        training, if `other_factor=1`, 100 cells that are not B cells 
        will be trained on with the B cells.

    Returns
    -------
    train_idx : dict
        Keys are cell classes and values are the train indices to 
        train scmkl that include both target and non-target samples.
    """
    uniq_labels = np.unique(cell_labels)
    train_idx = dict()

    if isinstance(cell_labels, pd.Series):
        cell_labels = cell_labels.to_numpy()
    elif isinstance(cell_labels, list):
        cell_labels = np.array(cell_labels)

    for lab in uniq_labels:
        target_pos = np.where(lab == cell_labels[train_indices])[0]
        overlap = np.isin(target_pos, train_indices)

        target_pos = target_pos[overlap]
        other_pos = np.setdiff1d(train_indices, target_pos)

        if (other_factor*target_pos.shape[0]) <= other_pos.shape[0]:
            n_samples = int(other_factor*target_pos.shape[0])
        else:
            n_samples = other_pos.shape[0]

        other_pos = seed_obj.choice(other_pos, n_samples, False)

        lab_train = np.concatenate([target_pos, other_pos])
        train_idx[lab] = lab_train.copy()

    return train_idx


def one_v_rest(adatas : list | ad.AnnData, names : list, 
               alpha_params : np.ndarray, tfidf : list=None, batches: int=10, 
               batch_size: int=100, train_dict: dict=None, 
               force_balance: bool=False, other_factor: float=1.0)-> dict:
    """
    For each cell class, creates model(s) comparing that class to all 
    others. Then, predicts on the training data using `scmkl.run()`.
    Only labels in both training and testing will be run.

    Parameters
    ----------
    adatas : list[AnnData]
        List of `ad.AnnData` objects created by `create_adata()` 
        where each `ad.AnnData` is one modality and composed of both 
        training and testing samples. Requires that `'train_indices'`
        and `'test_indices'` are the same between all `ad.AnnData`s.

    names : list[str]
        String variables that describe each modality respective to 
        `adatas` for labeling.
        
    alpha_params : np.ndarray | float | dict
        If is `dict`, expects keys to correspond to each unique label 
        with float as key (ideally would be the output of 
        scmkl.optimize_alpha). Else, array of alpha values to create 
        each model with or a float to run with a single alpha.

    tfidf : list[bool]
        If element `i` is `True`, `adatas[i]` will be TF-IDF 
        normalized. If `None`, no views will be TF-IDF normalized.

    batches : int
        The number of batches to use for the distance calculation. 
        This will average the result of `batches` distance calculations 
        of `batch_size` randomly sampled cells. More batches will 
        converge to population distance values at the cost of 
        scalability.

    batch_size : int
        The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability.
        If `batches*batch_size > num_training_cells`,
        `batch_size` will be reduced to 
        `int(num_training_cells / batches)`.

    force_balance : bool
        If `True`, training sets will be balanced to reduce class label 
        imbalance. Defaults to `False`.

    other_factor : float
        The ratio of cells to sample for the other class for each 
        model. For example, if classifying B cells with 100 B cells in 
        training, if `other_factor=1`, 100 cells that are not B cells 
        will be trained on with the B cells.

    Returns
    -------
    results : dict
        Contains keys for each cell class with results from cell class
        versus all other samples. See `scmkl.run()` for futher details. 
        Will also include a probablilities table with the predictions 
        from each model.

    Examples
    --------
    >>> adata = scmkl.create_adata(X = data_mat, 
    ...                            feature_names = gene_names, 
    ...                            group_dict = group_dict)
    >>>
    >>> results = scmkl.one_v_rest(adatas = [adata], names = ['rna'],
    ...                           alpha_list = np.array([0.05, 0.1]),
    ...                           tfidf = [False])
    >>>
    >>> adata.keys()
    dict_keys(['B cells', 'Monocytes', 'Dendritic cells', ...])
    """
    if isinstance(adatas, ad.AnnData):
        adatas = [adatas]
    if isinstance(tfidf, type(None)):
        tfidf = len(adatas)*[False]

    _check_adatas(adatas, check_obs=True, check_uns=True)

    # Want to retain all original train indices
    train_indices = adatas[0].uns['train_indices'].copy()
    test_indices = adatas[0].uns['test_indices'].copy()

    uniq_labels = _eval_labels(cell_labels = adatas[0].obs['labels'], 
                               train_indices = train_indices,
                               test_indices = test_indices)

    if (len(adatas) == 1) and ('Z_train' not in adatas[0].uns.keys()):
        adata = calculate_z(adatas[0], n_features = 5000, 
                            batches=batches, batch_size=batch_size)
    elif len(adatas) > 1:
        adata = multimodal_processing(adatas=adatas, 
                                      names=names, 
                                      tfidf=tfidf,
                                      batches=batches,
                                      batch_size=batch_size)
    else:
        adata = adatas[0]

    # Preventing multiple copies of adata(s) in memory
    del adatas
    gc.collect()

    # Need obj for capturing results
    results = dict()

    # Capturing cell labels to regenerate at each comparison
    cell_labels = np.array(adata.obs['labels'].copy())

    # Capturing perfect train/test splits for each class
    if train_dict:
        train_idx = train_dict
    else:
        if force_balance:
            train_idx = get_class_train(adata.uns['train_indices'], 
                                        cell_labels, 
                                        adata.uns['seed_obj'],
                                        other_factor)
    tracemalloc.start()
    for label in uniq_labels:

        print(f"Comparing {label} to other types", flush = True)
        cur_labels = cell_labels.copy()
        cur_labels[cell_labels != label] = 'other'

        # Need cur_label vs rest to run model
        adata.obs['labels'] = cur_labels

        if force_balance or train_dict:
            adata.uns['train_indices'] = train_idx[label]

        # Will only run scMKL with tuned alphas
        if isinstance(alpha_params, dict):
            alpha_list = np.array([alpha_params[label]])
        elif isinstance(alpha_params, float):
            alpha_list = np.array([alpha_params])
        else:
            alpha_list = alpha_params
        
        # Running scMKL
        results[label] = run(adata, alpha_list, return_probs=True)
        gc.collect()

    # Getting final predictions
    if isinstance(alpha_params, dict):
        alpha = alpha_params
    else:
        alpha = np.min(alpha_params)

    prob_table, pred_class, low_conf = get_prob_table(results, alpha)
    macro_f1 = f1_score(cell_labels[adata.uns['test_indices']], 
                        pred_class, average='macro')

    model_summary = per_model_summary(results, uniq_labels, alpha)

    # Global adata obj will be permanently changed if not reset
    adata.obs['labels'] = cell_labels
    adata.uns['train_indices'] = train_indices

    # Need to document vars, probs, and stats
    results['Per_model_summary'] = model_summary
    results['Classes'] = uniq_labels
    results['Probability_table'] = prob_table
    results['Predicted_class'] = pred_class
    results['Truth_labels'] = cell_labels[adata.uns['test_indices']]
    results['Low_confidence'] = low_conf
    results['Macro_F1-Score'] = macro_f1

    if force_balance or train_dict:
        results['Training_indices'] = train_idx

    return results