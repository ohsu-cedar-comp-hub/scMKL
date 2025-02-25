import numpy as np
import pandas as pd


def get_summary(results : dict, metric = 'AUROC'):
    '''
    Takes the results from either `scmkl.run()` and generates a 
    dataframe for each model containing columns for alpha, area under 
    the ROC, number of groups with nonzero weights, and highest 
    weighted group.

    Parameters
    ----------
    **results** : *dict*
        > A dictionary of results from scMKL generated from either 
        `scmkl.run()`.

    **metric** : *str*
        > Which metric to include in the summary. Default is AUROC. 
        Options include `'AUROC'`, `'Recall'`, `'Precision'`, 
        `'Accuracy'`, and `'F1-Score'`.

    Returns
    -------
    **summary_df** : *pd.DataFrame*
        > A table with columns:
        `['Alpha', 'AUROC', 'Number of Selected Groups', 'Top Group']`.
    
    Examples
    --------
    >>> results = scmkl.run(adata, alpha_list)
    >>> summary_df = scmkl.get_summary(results)
    ...
    >>> summary_df.head()
        Alpha   AUROC  Number of Selected Groups 
    0   2.20  0.8600                          3   
    1   1.96  0.9123                          4   
    2   1.72  0.9357                          5   
    3   1.48  0.9524                          7   
    4   1.24  0.9666                          9   
        Top Group
    0   RNA-HALLMARK_E2F_TARGETS
    1   RNA-HALLMARK_ESTROGEN_RESPONSE_EARLY
    2   RNA-HALLMARK_ESTROGEN_RESPONSE_EARLY
    3   RNA-HALLMARK_ESTROGEN_RESPONSE_EARLY
    4   RNA-HALLMARK_ESTROGEN_RESPONSE_EARLY
    '''
    summary = {'Alpha' : [],
           'AUROC' : [],
           'Number of Selected Groups' : [],
           'Top Group' : []}
    
    alpha_list = list(results['Metrics'].keys())

    # Creating summary DataFrame for each model
    for alpha in alpha_list:
        cur_alpha_rows = results['Norms'][alpha]
        top_weight_rows = np.max(results['Norms'][alpha])
        top_group_index = np.where(cur_alpha_rows == top_weight_rows)
        num_selected = len(results['Selected_groups'][alpha])
        top_group_names = np.array(results['Group_names'])[top_group_index]

        summary['Alpha'].append(alpha)
        summary['AUROC'].append(results['Metrics'][alpha][metric])
        summary['Number of Selected Groups'].append(num_selected)
        summary['Top Group'].append(*top_group_names)
    
    summary = pd.DataFrame(summary)

    return summary
