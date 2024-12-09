import numpy as np
import pandas as pd
import gc

from scmkl.run import run
from scmkl.estimate_sigma import estimate_sigma
from scmkl.calculate_z import calculate_z
from scmkl.multimodal_processing import multimodal_processing
from scmkl._checks import _check_adatas


def _eval_labels(cell_labels : np.ndarray, train_indices : np.ndarray, 
                  test_indices : np.ndarray) -> np.ndarray:
    '''
    Takes an array of multiclass cell labels and returns a unique array 
    of cell labels to test for.

    Parameters
    ----------
    cell_labels : np.ndarray
        > Cell labels that coorespond to an AnnData object.

    train_indices : np.ndarray
        > Indices for the training samples in an AnnData object.
    
    test_indices - np.ndarray
        > Indices for the testing samples in an AnnData object.

    remove_labels : bool
        > If True, models will only be created for cell labels in both
        the training and test data, if False, models will be generated
        for all cell labels in the training data.

    Returns
    -------
    uniq_labels : np.ndarray
        > Returns a numpy array of unique cell labels to be iterated 
        through during one versus all setups.
    '''
    train_uniq_labels = np.unique(cell_labels[train_indices])
    test_uniq_labels = np.unique(cell_labels[test_indices])

    # Getting only labels in both training and testing sets
    uniq_labels = np.intersect1d(train_uniq_labels, test_uniq_labels)

    # Ensuring that at least one cell type label between the two data
    #   are the same
    assert len(np.intersect1d(train_uniq_labels, test_uniq_labels)) > 0, \
        "There are no common labels between cells in the training and \
            testing samples"

    return uniq_labels


def _prob_table(results : dict, alpha):
    '''
    Takes a results dictionary with class and probabilities keys and 
    returns a table of probabilities for each class and the most 
    probable class for each cell.

    Parameters
    ----------
    results : dict
        > A nested dictionary that contains a dictionary for each class 
        containing probabilities for each cell class.

    alpha : float
        > A float for which model probabilities should be evaluated 
        for.

    Returns
    -------
    prob_table : pd.DataFrame
        > Each column is a cell class and the elements are the
        class probability outputs from the model.

    pred_class : list
        > The most probable cell classes respective to the training set 
        cells. 
    '''
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


def one_v_rest(adatas : list, names : list, alpha_list : np.ndarray, 
              tfidf : list) -> dict:
    '''
    For each cell class, creates model(s) comparing that class to all 
    others. Then, predicts on the training data using `scmkl.run()`.
    Only labels in both training and testing will be run.

    Parameters
    ----------
    **adatas** : *list[AnnData]* 
        > List of AnnData objects created by create_adata()
        where each AnnData is one modality and composed of both 
        training and testing samples. Requires that `'train_indices'`
        and `'test_indices'` are the same across all AnnDatas.

    **names** : *list[str]* 
        > List of string variables that describe each modality
        respective to adatas for labeling.
        
    **alpha_list** : *np.ndarray* | *float*
        > An array of alpha values to create each model with.

    **tfidf** : *list[bool]* 
        > List where if element i is `True`, adata[i] will be TFIDF 
        normalized.

    Returns
    -------
    **results** : *dict*
    > Contains keys for each cell class with results from cell class
    versus all other samples. See `scmkl.run()` for futher details.

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
                                 test_indices = test_indices)


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
        results[label] = run(adata, alpha_list, return_probs = True)

    # Getting final predictions
    alpha = np.min(alpha_list)
    prob_table, pred_class, low_conf = _prob_table(results, alpha)

    results['Probability_table'] = prob_table
    results['Predicted_class'] = pred_class
    results['Truth_labels'] = cell_labels[adata.uns['test_indices']]
    results['Low_confidence'] = low_conf

    return results