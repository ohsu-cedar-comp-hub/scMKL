import os
import re
import numpy as np
import pandas as pd


def _parse_result_type(results: dict):
    """
    Returns whether or not there are multiple results per class.

    Parameters
    ----------
    results : dict
        Either the output of `scmkl.run()` or `scmkl.one_v_rest()` or 
        a dictionary of those results.

    Returns
    -------
    is_mult, is_many : bool, bool
        If `is_mult` is `True`, then results are multiclass. If 
        `is_many` is `True`, results contain multiple outputs.

    """
    # Single result cases
    if 'Classes' in results.keys():
        is_mult = True
        is_many = False
        return is_mult, is_many
    elif 'Norms' in results.keys():
        is_mult = False
        is_many = False
        return is_mult, is_many

    # Multiresult cases
    keys = list(results.keys())
    if 'Classes' in results[keys[0]].keys():
        is_mult = True
        is_many = True
        return is_mult, is_many
    elif 'Norms' in results[keys[0]].keys():
        is_mult = False
        is_many = True
        return is_mult, is_many
    else:
        print("Unknown result structure", flush=True)


def sort_groups(df: pd.DataFrame, group_col: str='Group', 
                norm_col: str='Kernel Weight'):
    """
    Takes a dataframe with `group_col` and returns sorted group list 
    with groups in decending order by their weights. Assumes there is 
    one instance of each group.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with `group_col` and `norm_col` to be sorted by.

    group_col : str
        The column containing the group names.

    norm_col : str
        The column containing the kernel weights.

    Returns
    -------
    group_order : list
        A list of groups in descending order according to their kernel 
        weights.

    Examples
    --------
    >>> result = scmkl.run(adata, alpha_list)
    >>> weights = scmkl.get_weights(result)
    >>> group_order = scmkl.sort_groups(weights, 'Group', 
    ...                                 'Kernel Weight')
    >>>
    >>> group_order
    ['HALLMARK_ESTROGEN_RESPONSE_EARLY', 'HALLM...', ...]
    """
    df = df.copy()
    df = df.sort_values(norm_col, ascending=False)
    group_order = list(df[group_col])

    return group_order


def format_group_names(group_names: list | pd.Series | np.ndarray, 
                       rm_words: list=list()):
    """
    Takes an ArrayLike object of group names and formats them.

    Parameters
    ----------
    group_names : array_like
        An array of group names to format.

    rm_words : list
        Words to remove from all group names.

    Returns
    -------
    new_group_names : list
        Formatted version of the input group names.

    Examples
    --------
    >>> groups = ['HALLMARK_E2F_TARGETS', 'HALLMARK_HYPOXIA']
    >>> new_groups = scmkl.format_group_names(groups)
    >>> new_groups
    ['Hallmark E2F Targets', 'Hallmark Hypoxia']
    """
    new_group_names = list()
    rm_words = [word.lower() for word in rm_words]

    for name in group_names:
        new_name = list()
        for word in re.split(r'_|\s', name):
            if word.isalpha() and (len(word) > 3):
                word = word.capitalize()
            if word.lower() not in rm_words:
                new_name.append(word)
        new_name = ' '.join(new_name)
        new_group_names.append(new_name)

    return new_group_names
        

def parse_metrics(results: dict, key: str | None=None, 
                   include_as: bool=False) -> pd.DataFrame:
    """
    This function returns a pd.DataFrame for a single scMKL result 
    with performance results.

    Parameters
    ----------
    results : dict
        A result dictionary from `scmkl.run()`.
    
    key : str
        If specified, will add a key column to the output dataframe 
        where each element is `key`.

    include_as : bool
        If `True`, will add a column indicating which models' used 
        the optimal alphas.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with columns `['Alpha', 'Metric', 'Value']`. 
        `'Key'` col only added if `key` is not `None`.
    """
    df = {
        'Alpha' : list(),
        'Metric' : list(),
        'Value' : list()
    }

    # Check if is a multiclass result
    is_mult, _ = _parse_result_type(results)

    if is_mult:
        df['Class'] = list()

    # Ensuring results is a scMKL result and checking multiclass
    if 'Metrics' in results.keys():
        for alpha in results['Metrics'].keys():
            for metric, value in results['Metrics'][alpha].items():
                df['Alpha'].append(alpha)
                df['Metric'].append(metric)
                df['Value'].append(value)

    elif 'Classes' in results.keys():
        for ct in results['Classes']:
            for alpha in results[ct]['Metrics'].keys():
                for metric, value in results[ct]['Metrics'][alpha].items():
                    df['Alpha'].append(alpha)
                    df['Metric'].append(metric)
                    df['Value'].append(value)
                    df['Class'].append(ct)

    else:
        print(f"{key} is not a scMKL result and will be ignored.")
            
    df = pd.DataFrame(df)
    
    if include_as:
        assert 'Alpha_star' in results.keys(), "'Alpha_star' not in results"
        df['Alpha Star'] = df['Alpha'] == results['Alpha_star']

    if key is not None:
        df['Key'] = [key] * df.shape[0]

    return df        


def parse_weights(results: dict, include_as: bool=False, 
                   key: None | str=None) -> pd.DataFrame:
    """
    This function returns a pd.DataFrame for a single scMKL result 
    with group weights.

    Parameters
    ----------
    results : dict
        A result dictionary from `scmkl.run()`.
    
    key : str
        If specified, will add a key column to the output dataframe 
        where each element is `key`.

    include_as : bool
        If `True`, will add a column indicating which models' used 
        the optimal alphas.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with columns `['Alpha', 'Group', 
        'Kernel Weight']`. `'Key'` col only added if `key` is not 
        `None`.
    """
    df = {
        'Alpha' : list(),
        'Group' : list(),
        'Kernel Weight' : list()
    }

    # Check if is a multiclass result
    is_mult, _ = _parse_result_type(results)

    if is_mult:
        df['Class'] = list()

    # Ensuring results is a scMKL result and checking multiclass
    if 'Norms' in results.keys():
        for alpha in results['Norms'].keys():
            df['Alpha'].extend([alpha]*len(results['Norms'][alpha]))
            df['Group'].extend(results['Group_names'])
            df['Kernel Weight'].extend(results['Norms'][alpha])

    elif 'Classes' in results.keys():
        for ct in results['Classes']:
            for alpha in results[ct]['Norms'].keys():
                df['Alpha'].extend([alpha] * len(results[ct]['Norms'][alpha]))
                df['Group'].extend(results[ct]['Group_names'])
                df['Kernel Weight'].extend(results[ct]['Norms'][alpha])
                df['Class'].extend([ct]*len(results[ct]['Norms'][alpha]))

    df = pd.DataFrame(df)
    
    if include_as:
        df['Alpha Star'] = df['Alpha'] == results['Alpha_star'] 

    if key is not None:
        df['Key'] = [key] * df.shape[0]

    return df


def extract_results(results: dict, metric: str):
    """
    
    """
    summary = {'Alpha' : list(),
               metric : list(),
               'Number of Selected Groups' : list(),
               'Top Group' : list()}
    
    alpha_list = list(results['Metrics'].keys())

    # Creating summary DataFrame for each model
    for alpha in alpha_list:
        cur_alpha_rows = results['Norms'][alpha]
        top_weight_rows = np.max(results['Norms'][alpha])
        top_group_index = np.where(cur_alpha_rows == top_weight_rows)
        num_selected = len(results['Selected_groups'][alpha])
        top_group_name = np.array(results['Group_names'])[top_group_index]
        
        if 0 == num_selected:
            top_group_name = ["No groups selected"]

        summary['Alpha'].append(alpha)
        summary[metric].append(results['Metrics'][alpha][metric])
        summary['Number of Selected Groups'].append(num_selected)
        summary['Top Group'].append(*top_group_name)
    
    return pd.DataFrame(summary)


def get_summary(results: dict, metric: str='AUROC'):
    """
    Takes the results from `scmkl.run()` and generates a dataframe 
    for each model containing columns for alpha, area under the ROC, 
    number of groups with nonzero weights, and highest weighted 
    group.

    Parameters
    ----------
    results : dict
        A dictionary of results from scMKL generated from 
        `scmkl.run()`.

    metric : str
        Which metric to include in the summary. Default is AUROC. 
        Options include `'AUROC'`, `'Recall'`, `'Precision'`, 
        `'Accuracy'`, and `'F1-Score'`.

    Returns
    -------
    summary_df : pd.DataFrame
        A table with columns: `['Alpha', 'AUROC', 
        'Number of Selected Groups', 'Top Group']`.
    
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
    """
    is_multi, is_many = _parse_result_type(results)
    assert not is_many, "This function only supports single results"
    
    if is_multi:
        summaries = list()
        for ct in results['Classes']:
            data = extract_results(results[ct], metric)
            data['Class'] = [ct]*len(data)
            summaries.append(data.copy())
        summary = pd.concat(summaries)

    else:
        summary = extract_results(results, metric)

    return summary


def read_files(dir: str, pattern: str | None=None) -> dict:
    """
    This function takes a directory of scMKL results as pickle files 
    and returns a dictionary with the file names as keys and the data 
    from the respective files as the values.

    Parameters
    ----------
    dir : str
        A string specifying the file path for the output scMKL runs.

    pattern : str
        A regex string for filtering down to desired files. If 
        `None`, all files in the directory with the pickle file 
        extension will be added to the dictionary.

    Returns
    -------
    results : dict
        A dictionary with the file names as keys and data as values.

    Examples
    --------
    >>> filepath = 'scMKL_results/rna+atac/'
    ...
    >>> all_results = scmkl.read_files(filepath)
    >>> all_results.keys()
    dict_keys(['Rep_1.pkl', Rep_2.pkl, Rep_3.pkl, ...])
    """
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


def get_metrics(results: dict, include_as: bool=False) -> pd.DataFrame:
    """
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
    results : dict | None
        A dictionary with the results of a single run from 
        `scmkl.run()`. Must be `None` if `rfiles is not None`.

    rfiles : dict | None
        A dictionary of results dictionaries containing multiple 
        results from `scmkl.run()`. 

    include_as : bool
        When `True`, will add a bool col to output pd.DataFrame 
        where rows with alphas cooresponding to alpha_star will be 
        `True`.

    Returns
    -------
    df : pd.DataFrame
        A pd.DataFrame containing all of the metrics present from 
        the runs input.

    Examples
    --------
    >>> # For a single file
    >>> results = scmkl.run(adata)
    >>> metrics = scmkl.get_metrics(results = results)

    >>> # For multiple runs saved in a dict
    >>> output_dir = 'scMKL_outputs/'
    >>> rfiles = scmkl.read_files(output_dir)
    >>> metrics = scmkl.get_metrics(rfiles=rfiles)
    """
    # Checking which data is being worked with 
    is_mult, is_many = _parse_result_type(results)

    # Initiating col list with minimal columns
    cols = ['Alpha', 'Metric', 'Value']

    if include_as:
        cols.append('Alpha Star')
    if is_mult:
        cols.append('Class')

    if is_many:
        cols.append('Key')
        df = pd.DataFrame(columns = cols)
        for key, result in results.items():
                cur_df = parse_metrics(results = result, key = key, 
                                        include_as = include_as)
                df = pd.concat([df, cur_df.copy()])
            
    else:
        df = parse_metrics(results = results, include_as = include_as)

    return df


def get_weights(results : dict, include_as : bool = False) -> pd.DataFrame:
    """
    Takes either a single scMKL result or dictionary of results and 
    returns a pd.DataFrame with cols ['Alpha', 'Group', 
    'Kernel Weight']. If `include_as == True`, a fourth col will be 
    added to indicate whether or not the run respective to that alpha 
    was chosen as optimal via cross validation.

    Parameters
    ----------
    results : dict | None
        A dictionary with the results of a single run from 
        `scmkl.run()`. Must be `None` if `rfiles is not None`.

    rfiles : dict | None
        A dictionary of results dictionaries containing multiple 
        results from `scmkl.run()`. 

    include_as : bool
        When `True`, will add a bool col to output pd.DataFrame 
        where rows with alphas cooresponding to alpha_star will be 
        `True`.

    Returns
    -------
    df : pd.DataFrame
        A pd.DataFrame containing all of the groups from each alpha 
        and their cooresponding kernel weights.

    Examples
    --------
    >>> # For a single file
    >>> results = scmkl.run(adata)
    >>> weights = scmkl.get_weights(results = results)
    
    >>> # For multiple runs saved in a dict
    >>> output_dir = 'scMKL_outputs/'
    >>> rfiles = scmkl.read_files(output_dir)
    >>> weights = scmkl.get_weights(rfiles=rfiles)
    """
    # Checking which data is being worked with 
    is_mult, is_many = _parse_result_type(results)

    # Initiating col list with minimal columns
    cols = ['Alpha', 'Group', 'Kernel Weight']

    if include_as:
        cols.append('Alpha Star')
    if is_mult:
        cols.append('Class')

    if is_many:
        cols.append('Key')
        df = pd.DataFrame(columns = cols)
        for key, result in results.items():
            cur_df = parse_weights(results = result, key = key, 
                                     include_as = include_as)
            df = pd.concat([df, cur_df.copy()])
            
    else:
        df = parse_weights(results = results, include_as = include_as)

    return df


def get_selection(weights_df: pd.DataFrame, 
                  order_groups: bool=False) -> pd.DataFrame:
    """
    This function takes a pd.DataFrame created by 
    `scmkl.get_weights()` and returns a selection table. Selection 
    refers to how many times a group had a nonzero group weight. To 
    calculate this, a col is added indicating whether the group was 
    selected. Then, the dataframe is grouped by alpha and group. 
    Selection can then be summed returning a dataframe with cols 
    `['Alpha', 'Group', Selection]`. If is the result of multiclass 
    run(s), `'Class'` column must be present and will be in resulting 
    df as well.

    Parameters
    ----------
    weights_df : pd.DataFrame
        A dataframe output by `scmkl.get_weights()` with cols
        `['Alpha', 'Group', 'Kernel Weight']`. If is the result of 
        multiclass run(s), `'Class'` column must be present as well.

    order_groups : bool
        If `True`, the `'Group'` col of the output dataframe will be 
        made into a `pd.Categorical` col ordered by number of times 
        each group was selected in decending order.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with cols `['Alpha', 'Group', Selection]`. Also, 
        `'Class'` column if is a multiclass result.

    Example
    -------
    >>> # For a single file
    >>> results = scmkl.run(adata)
    >>> weights = scmkl.get_weights(results = results)
    >>> selection = scmkl.get_selection(weights)
    
    >>> # For multiple runs saved in a dict
    >>> output_dir = 'scMKL_outputs/'
    >>> rfiles = scmkl.read_files(output_dir)
    >>> weights = scmkl.get_weights(rfiles=rfiles)
    >>> selection = scmkl.get_selection(weights)
    """
    # Adding col indicating whether or not groups have nonzero weight
    selection = weights_df['Kernel Weight'].apply(lambda x: x > 0)
    weights_df['Selection'] = selection

    # Summing selection across replications to get selection
    is_mult = 'Class' in weights_df.columns
    if is_mult:
        df = weights_df.groupby(['Alpha', 'Group', 'Class'])['Selection'].sum()
    else:
        df = weights_df.groupby(['Alpha', 'Group'])['Selection'].sum()
    df = df.reset_index()

    # Getting group order
    if order_groups and not is_mult:
        order = df.groupby('Group')['Selection'].sum()
        order = order.reset_index().sort_values(by = 'Selection', 
                                                ascending = False)
        order = order['Group']
        df['Group'] = pd.Categorical(df['Group'], categories = order)


    return df


def groups_per_alpha(selection_df: pd.DataFrame) -> dict:
    """
    This function takes a pd.DataFrame from `scmkl.get_selection()` 
    generated from multiple scMKL results and returns a dictionary 
    with keys being alphas from the input dataframe and values being 
    the mean number of selected groups for a given alpha across 
    results. 

    Parameters
    ----------
    selection_df : pd.DataFrame
        A dataframe output by `scmkl.get_selection()` with cols 
        `['Alpha', 'Group', Selection].
    
    Returns
    -------
    mean_groups : dict
        A dictionary with alphas as keys and the mean number of 
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
    """
    mean_groups = {}
    for alpha in np.unique(selection_df['Alpha']):

        # Capturing rows for given alpha
        rows = selection_df['Alpha'] == alpha

        # Adding mean number of groups for alpha
        mean_groups[alpha] = np.mean(selection_df[rows]['Selection'])

    return mean_groups


def read_gtf(path: str, filter_to_coding: bool=False):
    """
    Reads and formats a gtf file. Adds colnames: `['chr', 'source', 
    'feature', 'start', 'end', 'score', 'strand', 'frame', 
    'attribute']`.

    Parameters
    ----------
    path : str
        The file path to the gtf file to be read in. If the file is 
        gzipped, file name must end with .gz.

    filter_to_coding : bool
        If `True`, will filter rows in gtf data frame to only 
        protein coding genes. Will add column `'gene_name'` containing 
        the gene name for each row.

    Returns
    -------
    df : pd.DataFrame
        A pandas dataframe of the input gtf file.

    Examples
    --------
    >>> import scmkl
    >>>
    >>> file = 'data/hg38_subset_protein_coding.annotation.gtf'
    >>> gtf = scmkl.read_gtf(file)
    >>>
    >>> gtf.head()
            chr  source     feature  start    end score strand frame                                          
    0  chr1  HAVANA        gene  11869  14409     .      +     .  
    1  chr1  HAVANA  transcript  11869  14409     .      +     .  
    2  chr1  HAVANA        exon  11869  12227     .      +     .  
    3  chr1  HAVANA        exon  12613  12721     .      +     .  
    4  chr1  HAVANA        exon  13221  14409     .      +     .  
    attribute
    gene_id "ENSG00000223972.5"; gene_type "transc...
    gene_id "ENSG00000223972.5"; transcript_id "EN...
    gene_id "ENSG00000223972.5"; transcript_id "EN...
    gene_id "ENSG00000223972.5"; transcript_id "EN...
    gene_id "ENSG00000223972.5"; transcript_id "EN...
    """
    df = pd.read_csv(path, sep='\t', comment='#', 
                     skip_blank_lines=True, header=None)
    
    df.columns = ['chr', 'source', 'feature', 'start', 'end', 
                  'score', 'strand', 'frame', 'attribute']
    
    if filter_to_coding:
        prot_rows = df['attribute'].str.contains('protein_coding')
        df = df[prot_rows]
        df = df[df['feature'] == 'gene']

        # Capturing and adding gene name to df
        df['gene_name'] = [re.findall(r'(?<=gene_name ")[A-z0-9]+', 
                                      attr)[0] 
                           for attr in df['attribute']]
    
    return df