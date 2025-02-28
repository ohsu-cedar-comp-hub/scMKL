import os
import re
import numpy as np
import pandas as pd


def _parse_result_type(results : dict | None, rfiles : dict | None) -> bool:
    '''
    This function simply returns a bool for whether or not there are 
    multiple runs present while checking that There is one dict and 
    one Nonetype between `results` and `rfiles`.
    '''
    dtypes = (type(results), type(rfiles))
    none_in_dtypes = type(None) in dtypes
    dict_in_dtypes = dict in dtypes
    both_in_dtypes = none_in_dtypes and dict_in_dtypes

    # Ensuring that at least one of dtypes is None
    assert both_in_dtypes, "Only `rfiles` or `results` can be provided"

    if type(rfiles) is dict:
        mult_files = True
    else:
        mult_files = False

    return mult_files


def _parse_metrics(results, key : str | None = None, 
                   include_as = False) -> pd.DataFrame:
    '''
    This function returns a pd.DataFrame for a single scMKL result.
    '''
    alpha_vals = []
    met_names = []
    met_vals = []

    for alpha in results['Metrics'].keys():
        for metric, value in results['Metrics'][alpha].items():
            alpha_vals.append(alpha)
            met_names.append(metric)
            met_vals.append(value)
            
    df = pd.DataFrame({'Alpha' : alpha_vals,
                       'Metric' : met_names,
                       'Value' : met_vals})
    
    if include_as:
        assert 'Alpha_star' in results.keys(), "'Alpha_star' not in results"
        df['Alpha Star'] = df['Alpha'] == results['Alpha_star']

    if key is not None:
        df['Key'] = [key] * df.shape[0]

    return df            


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


def read_files(dir : str, pattern : str | None = None) -> dict:
    '''
    This function takes a directory of scMKL results as pickle files 
    and returns a dictionary with the file names as keys and the data 
    from the respective files as the values.

    Parameters
    ----------
    **dir** : *str*
        > A string specifying the file path for the output scMKL runs.

    **pattern** : *str*
        > A regex string for filtering down to desired files. If 
        `None`, all files in the directory with the pickle file 
        extension will be added to the dictionary.

    Returns
    -------
    **results** : *dict*
        > a dictionary with the file names as keys and data as values.

    Examples
    --------
    >>> filepath = 'scMKL_results/rna+atac/'
    ...
    >>> all_results = scmkl.read_files(filepath)
    >>> all_results.keys()
    dict_keys(['Rep_1.pkl', Rep_2.pkl, Rep_3.pkl, ...])
    '''
    # Reading all pickle files in patter is None
    if pattern is None:
        data = {file : np.load(f'{dir}/{file}', allow_pickle = True)
                 for file in os.listdir(dir) if '.pkl' in file}
    
    # Reading only files matching pattern if not None
    else:
        pattern = repr(pattern)
        data = {file : np.load(f'{dir}/{file}', allow_pickle = True)
                 for file in os.listdir(dir) 
                 if re.fullmatch(pattern, file) is not None}
        
    return data


def get_metrics(results : dict | None = None, rfiles : dict | None = None, 
                include_as : bool = False) -> pd.DataFrame:
    '''
    Takes either a single scMKL result or a dictionary where each 
    entry cooresponds to one result. Returns a dataframe with cols 
    ['Alpha', 'Metric', 'Value']. If `include_as == True`, another 
    col of booleans will be added to indicate whether or not the run 
    respective to that alpha was chosen as optimal via CV. If 
    `include_key == True`, another column will be added with the name 
    of the key to the respective file (only applicable with multiple 
    results).

    Parameters
    ----------
    **results** : *None* | *dict*
        > A dictionary with the results of a single run from 
        `scmkl.run()`. Must be `None` if `rfiles is not None`.

    **rfiles** : *None* | *dict*
        > A dictionary of results dictionaries containing multiple 
        results from `scmkl.run()`. If `include_keys == True`, a col 
        will be added to the output pd.DataFrame with the keys as 
        values cooresponding to each row.

    **include_as** : *bool*
        > When `True`, will add a bool col to output pd.DataFrame 
        where rows with alphas cooresponding to alpha_star will be 
        `True`.

    Returns
    -------
    **df** : *pd.DataFrame*
        > A pd.DataFrame containing all of the metrics present from 
        the runs input.

    Examples
    --------
    >>> # For a single file
    >>> results = scmkl.run(adata)
    >>> metrics = scmkl.get_metrics(results = results)

    >>> # For multiple runs saved in a dict
    >>> output_dir = 'scMKL_outputs/'
    >>> rfiles = scmkl.read_files(output_dir)
    >>> metrics = scmkl.get_metrics(rfiles)
    '''
    # Checking which data is being worked with 
    multi_results = _parse_result_type(results = results, rfiles = rfiles)

    # Initiating col list with minimal columns
    cols = ['Alpha', 'Metric', 'Value']

    if include_as:
        cols.append('Alpha Star')

    if multi_results:
        cols.append('Key')
        df = pd.DataFrame(columns = cols)
        for key, result in rfiles.items():
            cur_df = _parse_metrics(results = result, key = key, 
                                     include_as = include_as)
            df = pd.concat([df, cur_df.copy()])
            
    else:
        df = _parse_metrics(results = results, include_as = include_as)

    return df

