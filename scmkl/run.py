import anndata as ad
import numpy as np
import time
import tracemalloc

from scmkl.train_model import train_model
from scmkl.test import predict, find_selected_groups


def run(adata : ad.AnnData, alpha_list : np.ndarray, 
        metrics : list | None = None) -> dict:
    '''
    Wrapper function for training and test with multiple alpha values.
    Returns metrics, predictions, group weights, and resource usage.

    Parameters
    ----------
    **adata** : *AnnData* 
        > A processed *AnnData* with `'Z_train'`, `'Z_test'`, and 
        `'group_dict'` keys in `adata.uns`.
    
    **alpha_list** : *np.ndarray* 
        > `alpha` values to create models using. Alpha refers to the 
        penalty parameter in Group Lasso. Larger alphas force group 
        weights to shrink towards 0 while smaller alphas apply a lesser 
        penalty to kernal weights.

    **metrics** : *list[str]*
        > What metrics should be calculated on predictions. Options are 
        ['AUROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall']. When 
        set to `None`, all metrics are calculated.
    
    Returns
    -------
    **results** : *dict*
    > With keys and values: 

    > `'Metrics'` : a nested dictionary as `[alpha][metric]` = value.
    
    > `'Selected_groups'` : a dictionary as `[alpha]` = array of 
        groups with nonzero weights.

    > `'Norms'` : a dictionary as `[alpha]` = array of kernel weights
        for each group, order respective to 'Group_names'.

    > `'Predictions'` : a dictionary as `[alpha]` = predicted class
        respective to 'Observations' for that `alpha`.

    > `'Observed'` : an array of ground truth cell labels from the
        test set.

    > `'Test_indices'` : indices of samples respective to adata 
        used in the training set.

    > `'Group_names'` : an array of group names respective to each
        array in 'Norms'.

    > `'Model'` : a dictionary where `[alpha]` = Celer Group Lasso
        object for that `alpha`.

    > `'RAM_usage'` : memory usage after training models for each 
        `alpha`.

    Examples
    --------
    >>> results = scmkl.run(adata = adata, alpha_list = np.array([0.05, 0.1, 0.5]))
    >>> results
    dict_keys(['Metrics', 'Selected_groups', 'Norms', 'Predictions', 'Observed', 
    'Test_indices', 'Group_names', 'Models', 'Train_time', 'RAM_usage'])
    >>>
    >>> # List of alpha values
    >>> results['Metrics'].keys()
    dict_keys([0.05, 0.1, 0.5])
    >>>
    >>> results['Metrics'][0.05]
    {'AUROC': 0.9859,
    'Accuracy': 0.945,
    'F1-Score': 0.9452736318407959,
    'Precision': 0.9405940594059405,
    'Recall': 0.95}
    '''
    if metrics == None:
        metrics = ['AUROC', 'F1-Score','Accuracy', 'Precision', 'Recall']

    # Initializing variables to capture metrics
    group_names = list(adata.uns['group_dict'].keys())
    predicted = {}
    group_norms = {}
    metric_dict = {}
    selected_groups = {}
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
        selected_groups[alpha] = find_selected_groups(adata)

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
    results['Selected_groups'] = selected_groups
    results['Norms'] = group_norms
    results['Predictions'] = predicted
    results['Observed'] = adata.obs['labels'].iloc[adata.uns['test_indices']]
    results['Test_indices'] = adata.uns['test_indices']
    results['Group_names']= group_names
    results['Models'] = models
    results['Train_time'] = train_time
    results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'

    return results