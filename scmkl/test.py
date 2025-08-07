import numpy as np
import sklearn.metrics as skm
import anndata as ad


def predict(adata: ad.AnnData, metrics: list | None=None,
            return_probs: bool=False):
    """
    Function to return predicted labels and calculate any of AUROC, 
    Accuracy, F1 Score, Precision, Recall for a classification. 

    **If labeled_test flag in `adata` is set to `False`,
    metrics cannot be computed.**
    
    Parameters
    ----------
    adata : ad.AnnData
        Has keys `'model'`, `'Z_train'`, and `'Z_test'` in `adata.uns`.

    metrics : list[str] | None
        Which metrics to calculate on the predicted values. Options
        are `'AUROC'`, `'Accuracy'`, `'F1-Score'`, `'Precision'`, and 
        `'Recall'`. If `None`, all five metrics are calculated.

    return_probs : bool
        If `True`, will return a dictionary with class probabilities.

    Returns
    -------
    y_pred : np.ndarray
        Predicted cell classes.

    metrics_dict : dict
        Contains `'AUROC'`, `'Accuracy'`, `'F1-Score'`, 
        `'Precision'`, and/or `'Recall'` keys depending on metrics 
        argument.

    probs : dict
        If `return_probs` is `True`, will return a dictionary with 
        probabilities for each class in `y_test`.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata = scmkl.calculate_z(adata)
    >>> metrics = ['AUROC', 'F1-Score', 'Accuracy', 'Precision', 
    ...            'Recall']
    >>> adata = scmkl.train_model(adata, metrics = metrics)
    >>>
    >>> metrics_dict = scmkl.predict(adata)
    >>> metrics_dict.keys()
    dict_keys(['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
    """
    X_test = adata.uns['Z_test']

    allowed_mets = ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']

    # Asserting all input metrics are valid
    if metrics is not None:
        mets_allowed = [metric in allowed_mets for metric in metrics]
        assert all(mets_allowed), ("Unknown metric provided. Must be None, "
                                   f"or one or more of {allowed_mets}")

    # Capturing class labels
    train_idx = adata.uns['train_indices']
    classes = np.unique(adata.obs['labels'].iloc[train_idx].to_numpy())

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1/(1 + np.exp(-adata.uns['model'].predict(X_test)))

    #Convert numerical probabilities into binary phenotype
    y_pred = np.array(np.repeat(classes[1], X_test.shape[0]), 
                      dtype = 'object')
    y_pred[np.round(probabilities, 0).astype(int) == 1] = classes[0]

    if not adata.uns['labeled_test']:
        if not metrics is None:
            print("WARNING: Cannot calculate classification metrics "
                  "for unlabeled test data")
            metrics = None
    else:
        y_test = adata.obs['labels'].iloc[adata.uns['test_indices']]
        y_test = y_test.to_numpy()
        X_test = adata.uns['Z_test']
        assert X_test.shape[0] == len(y_test), ("X rows and length of y must "
                                                "be equal")

        # Group Lasso requires 'continous' y values need to re-descritize it
        y = np.zeros((len(y_test)))
        y[y_test == classes[0]] = 1

        metric_dict = {}

        if (metrics is None) and (return_probs == False):
            return y_pred
        
        # Calculate and save metrics given in metrics
        p_cl = classes[0]
        if 'AUROC' in metrics:
            fpr, tpr, _ = skm.roc_curve(y, probabilities)
            metric_dict['AUROC'] = skm.auc(fpr, tpr)
        if 'Accuracy' in metrics:
            metric_dict['Accuracy'] = np.mean(y_test == y_pred)
        if 'F1-Score' in metrics:
            metric_dict['F1-Score'] = skm.f1_score(y_test, y_pred, 
                                                   pos_label = p_cl)
        if 'Precision' in metrics:
            metric_dict['Precision'] = skm.precision_score(y_test, y_pred, 
                                                           pos_label = p_cl)
        if 'Recall' in metrics:
            metric_dict['Recall'] = skm.recall_score(y_test, y_pred, 
                                                     pos_label = p_cl)

    if return_probs:
        probs = {classes[0] : probabilities,
                 classes[1] : 1 - probabilities}
        if metrics is not None:
            return y_pred, metric_dict, probs
        else:
            return y_pred, probs
    else:
        if metrics is not None:
            return y_pred, metric_dict
        else:
            return y_pred


def find_selected_groups(adata: ad.AnnData) -> np.ndarray:
    """
    Find feature groups selected by the model during training. If 
    feature weight assigned by the model is non-0, then the group 
    containing that feature is selected.

    Parameters
    ----------
    adata : ad.AnnData
        Has `celer.GroupLasso` object in `adata.uns['model']`.

    Returns
    -------
    selected_groups : np.ndarray
        Array containing the names of the groups with nonzero kernel 
        weights.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata = scmkl.calculate_z(adata)
    >>> adata = scmkl.train_model(adata)
    >>>
    >>> selected_groups = scmkl.find_selected_groups(adata)
    >>> selected_groups
    np.ndarray(['HALLMARK_ESTROGEN_RESPONSE_EARLY', 
                'HALLMARK_HYPOXIA'])
    """

    selected_groups = []
    coefficients = adata.uns['model'].coef_
    group_size = adata.uns['model'].get_params()['groups']
    group_names = np.array(list(adata.uns['group_dict'].keys()))

    # Loop over the model weights associated with each group and calculate 
    # the L2 norm
    for i, group in enumerate(group_names):
        if not isinstance(group_size, (list, set, np.ndarray, tuple)):
            group_start = i * group_size
            group_end = (i+1) * group_size - 1
            group_cols = np.arange(group_start, group_end)
            group_norm = np.linalg.norm(coefficients[group_cols])
        else: 
            group_norm = np.linalg.norm(coefficients[group_size[i]])

        # Only include the group if the model weights are > 0 
        if group_norm != 0:
            selected_groups.append(group)

    return np.array(selected_groups)
