import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import gc

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


def get_prob_table(results : dict, alpha: float):
    """
    Takes a results dictionary with class and probabilities keys and 
    returns a table of probabilities for each class and the most 
    probable class for each cell.

    Parameters
    ----------
    results : dict
        A nested dictionary that contains a dictionary for each class 
        containing probabilities for each cell class.

    alpha : float
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
    prob_table = {class_ : results[class_]['Probabilities'][alpha][class_]
                  for class_ in results.keys()}
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

    alpha : float
        The alpha for creating the summary from.

    Returns
    -------
    summary_df : pd.DataFrame
        Dataframe with classes on rows and metrics as cols.
    """
    # Getting metrics availible in results
    avail_mets = list(results[uniq_labels[0]]['Metrics'][alpha])

    summary_df = {metric : list()
                  for metric in avail_mets}
    summary_df['Class'] = uniq_labels

    for lab in summary_df['Class']:
        for met in avail_mets:
            val = results[lab]['Metrics'][alpha][met]
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
    uniq_labels = set(cell_labels)
    train_idx = dict()

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


def one_v_rest(adatas : list, names : list, alpha_list : np.ndarray, 
              tfidf : list, batches: int=10, batch_size: int=100, 
              force_balance: bool=False, other_factor: float=1.0) -> dict:
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
        
    alpha_list : np.ndarray | float
        An array of alpha values to create each model with or a float 
        to run with a single alpha.

    tfidf : list[bool]
        If element `i` is `True`, `adatas[i]` will be TF-IDF 
        normalized.

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
    # Formatting checks ensuring all adata elements are 
    # AnnData objects and train/test indices are all the same
    _check_adatas(adatas, check_obs = True, check_uns = True)

    # Extracting train and test indices
    train_indices = adatas[0].uns['train_indices']
    test_indices = adatas[0].uns['test_indices']

    # Checking and capturing cell labels
    uniq_labels = _eval_labels(cell_labels = adatas[0].obs['labels'], 
                               train_indices = train_indices,
                               test_indices = test_indices)


    # Calculating Z matrices, method depends on whether there are multiple 
    # adatas (modalities)
    if (len(adatas) == 1) and ('Z_train' not in adatas[0].uns.keys()):
        adata = calculate_z(adata, n_features = 5000, batches=batches, batch_size=batch_size)
    elif len(adatas) > 1:
        adata = multimodal_processing(adatas = adatas, 
                                      names = names, 
                                      tfidf = tfidf,
                                      batches=batches,
                                      batch_size=batch_size)
    else:
        adata = adatas[0].copy()

    del adatas
    gc.collect()

    # Initializing for capturing model outputs
    results = dict()

    # Capturing cell labels before overwriting
    cell_labels = np.array(adata.obs['labels'])

    # Capturing perfect train/test splits for each class
    if force_balance:
        train_idx = get_class_train(adata.uns['train_indices'], 
                                    cell_labels, 
                                    adata.uns['seed_obj'],
                                    other_factor)

    for label in uniq_labels:

        print(f"Comparing {label} to other types", flush = True)
        cur_labels = cell_labels.copy()
        cur_labels[cell_labels != label] = 'other'

        # Replacing cell labels for current cell type vs rest
        adata.obs['labels'] = cur_labels

        if force_balance:
            adata.uns['train_indices'] = train_idx[label]

        # Running scMKL
        results[label] = run(adata, alpha_list, return_probs = True)

    # Getting final predictions
    alpha = np.min(alpha_list)
    prob_table, pred_class, low_conf = get_prob_table(results, alpha)
    macro_f1 = f1_score(cell_labels[adata.uns['test_indices']], 
                        pred_class, average='macro')

    model_summary = per_model_summary(results, uniq_labels, alpha)

    results['Per_model_summary'] = model_summary
    results['Classes'] = uniq_labels
    results['Probability_table'] = prob_table
    results['Predicted_class'] = pred_class
    results['Truth_labels'] = cell_labels[adata.uns['test_indices']]
    results['Low_confidence'] = low_conf
    results['Macro_F1-Score'] = macro_f1

    if force_balance:
        results['Training_indices'] = train_idx

    return results