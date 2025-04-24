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

    # If statement ensuring results is a scMKL results with metrics
    if 'Metrics' in results.keys():
        for alpha in results['Metrics'].keys():
            for metric, value in results['Metrics'][alpha].items():
                alpha_vals.append(alpha)
                met_names.append(metric)
                met_vals.append(value)

    # Fix this for include_as parameter
    else:
        print(f"{key} is not a scMKL result and will be ignored.")
            
    df = pd.DataFrame({'Alpha' : alpha_vals,
                       'Metric' : met_names,
                       'Value' : met_vals})
    
    if include_as:
        assert 'Alpha_star' in results.keys(), "'Alpha_star' not in results"
        df['Alpha Star'] = df['Alpha'] == results['Alpha_star']

    if key is not None:
        df['Key'] = [key] * df.shape[0]

    return df        


def _parse_weights(results : dict, include_as : bool = False, 
                   key : None | str = None) -> pd.DataFrame:
    '''
    '''
    alpha_vals = []
    group_names = []
    kernel_weights = []

    for alpha in results['Norms'].keys():
        alpha_vals.extend([alpha] * len(results['Norms'][alpha]))
        group_names.extend(results['Group_names'])
        kernel_weights.extend(results['Norms'][alpha])

    df = pd.DataFrame({'Alpha' : alpha_vals, 
                       'Group' : group_names, 
                       'Kernel Weight' : kernel_weights})
    
    if include_as:
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
        summary[metric].append(results['Metrics'][alpha][metric])
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


def get_weights(results : dict | None = None, rfiles : dict | None = None, 
                include_as : bool = False) -> pd.DataFrame:
    '''
    Takes either a single scMKL result or dictionary of results and 
    returns a pd.DataFrame with cols ['Alpha', 'Group', 
    'Kernel Weight']. If include_as == True, a fourth col will be 
    added to indicate whether or not the run respective to that alpha 
    was chosen as optimal via CV.

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
        > A pd.DataFrame containing all of the groups from each alpha 
        and their cooresponding kernel weights.

    Examples
    --------
    >>> # For a single file
    >>> results = scmkl.run(adata)
    >>> weights = scmkl.get_weights(results = results)
    >>>
    >>> # For multiple runs saved in a dict
    >>> output_dir = 'scMKL_outputs/'
    >>> rfiles = scmkl.read_files(output_dir)
    >>> weights = scmkl.get_weights(rfiles)
    '''
    # Checking which data is being worked with 
    multi_results = _parse_result_type(results = results, rfiles = rfiles)

    # Initiating col list with minimal columns
    cols = ['Alpha', 'Group', 'Kernel Weight']

    if include_as:
        cols.append('Alpha Star')

    if multi_results:
        cols.append('Key')
        df = pd.DataFrame(columns = cols)
        for key, result in rfiles.items():
            cur_df = _parse_weights(results = result, key = key, 
                                     include_as = include_as)
            df = pd.concat([df, cur_df.copy()])
            
    else:
        df = _parse_metrics(results = results, include_as = include_as)

    return df


def get_selection(weights_df, order_groups : bool) -> pd.DataFrame:
    '''
    This function takes a pd.DataFrame created by 
    `scmkl.get_weights()` and returns a selection table. Selection 
    refers to how many times a group had a nonzero group weight. To 
    calculate this, a col is added indicating whether the group was 
    selected. Then, the dataframe is grouped by alpha and group. 
    Selection can then be summed returning a dataframe with cols 
    `['Alpha', 'Group', Selection]`.

    Parameters
    ----------
    **weights_df** : *pd.DataFrame*
        > A dataframe output by `scmkl.get_weights()` with cols
        `['Alpha', 'Group', 'Kernel Weight']`.

    **order_groups** : *bool*
        > If `True`, the `'Group'` col of the output dataframe will be 
        made into a `pd.Categorical` col ordered by number of times 
        each group was selected in decending order.

    Returns
    -------
    **df** : *pd.DataFrame*
        > A dataframe with cols `['Alpha', 'Group', Selection]`.

    Example
    -------
    >>> # For a single file
    >>> results = scmkl.run(adata)
    >>> weights = scmkl.get_weights(results = results)
    >>> selection = scmkl.get_selection(weights)
    >>>
    >>> # For multiple runs saved in a dict
    >>> output_dir = 'scMKL_outputs/'
    >>> rfiles = scmkl.read_files(output_dir)
    >>> weights = scmkl.get_weights(rfiles)
    >>> selection = scmkl.get_selection(weights)
    '''
    # Adding col indicating whether or not groups have nonzero weight
    selection = weights_df['Kernel Weight'].apply(lambda x: x > 0)
    weights_df['Selection'] = selection

    # Summing selection across replications to get selection
    df = weights_df.groupby(['Alpha', 'Group'])['Selection'].sum()
    df = df.reset_index()

    # Getting group order
    if order_groups:
        order = df.groupby('Group')['Selection'].sum()
        order = order.reset_index().sort_values(by = 'Selection', 
                                                ascending = False)
        order = order['Group']
        df['Group'] = pd.Categorical(df['Group'], categories = order)


    return df


def mean_groups_per_alpha(selection_df) -> dict:
    '''
    This function takes a pd.DataFrame from `scmkl.get_selection()` 
    generated from multiple scMKL results and returns a dictionary 
    with keys being alphas from the input dataframe and values being 
    the mean number of selected groups for a given alpha across 
    results. 

    Parameters
    ----------
    **selection_df** : *pd.DataFrame*
        > A dataframe output by `scmkl.get_selection()` with cols 
        `['Alpha', 'Group', Selection].
    
    Returns
    -------
    **mean_groups** : *dict*
        > A dictionary with alphas as keys and the mean number of 
        selected groups for that alpha as keys.

    Examples
    --------
    >>> weights = scmkl.get_weights(rfiles)
    >>> selection = scmkl.get_selection(weights)
    >>> mean_groups = scmkl.mean_groups_per_alpha(selection)
    >>> mean_groups = {alpha : np.round(num_selected, 1)
    ...                for alpha, num_selected in mean_groups.items()}
    >>>
    >>> print(mean_groups)
    {0.05 : 50.0, 0.2 : 24.7, 1.1 : 5.3}
    '''
    mean_groups = {}
    for alpha in np.unique(selection_df['Alpha']):

        # Capturing rows for given alpha
        rows = selection_df['Alpha'] == alpha

        # Adding mean number of groups for alpha
        mean_groups[alpha] = np.mean(selection_df[rows]['Selection'])

    return mean_groups