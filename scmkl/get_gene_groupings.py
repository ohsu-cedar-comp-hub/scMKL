import pickle
import numpy as np
import pandas as pd
import gseapy as gp


# Prevent iteration of all availible gene sets for human and mouse
global_lib_orgs = ['human', 'mouse']

human_genesets = [
    'Azimuth_2023', 
    'Azimuth_Cell_Types_2021', 
    'Cancer_Cell_Line_Encyclopedia', 
    'CellMarker_2024', 
    'CellMarker_Augmented_2021', 
    'GO_Biological_Process_2025', 
    'GO_Cellular_Component_2025', 
    'GO_Molecular_Function_2025', 
    'KEGG_2019_Mouse', 
    'KEGG_2021_Human', 
    'MSigDB_Hallmark_2020', 
    'NCI-60_Cancer_Cell_Lines', 
    'WikiPathways_2024_Human', 
    'WikiPathways_2024_Mouse'
]


def check_organism(organism: str):
    """
    Makes sure that organism is in availible organisms from `gseapy`.

    Parameters
    ----------
    organism : str
        The organsim to check.

    Returns
    -------
    is_in : bool
        `True` if the organism is valid. `False` if organism is not an 
        option.
    """
    org_opts = {'human', 'mouse', 'yeast', 'fly', 'fish', 'worm'}
    org_err = f"Invalid `organism`, choose from {org_opts}"
    assert organism.lower() in org_opts, org_err

    return None


def check_groups(groups: list, key_terms: str | list='', 
                 blacklist: str | list | bool=False, other_org: str=''):
    """
    Takes a list of groups from a gene set library and checks the names 
    for the desired gene sets. Returns a dictionary with keys 
    `'key_terms_in'` that is `list` of `bool`s corresponding to 
    `names`. `'num_groups'` value is an int for how many groups are in 
    the library.

    Parameters
    ----------
    groups : list
        The names of groups for a given library.

    key_terms : str | list
        The types of cells or other specifiers the gene set is for 
        (example: 'CD4 T', 'kidney', ect...).

    other_org : str
        Either `'human'` or `'mouse'`. The organsim to ignore when 
        checking groups. Should be empty if target organism is not 
        `'human'` or `'mouse'`.

    Returns
    -------
    result : dict
        A dictionary with the following keys and values:

        `'name'` : list
            The names of each group in input `'groups'` if the name 
            does not contain `'other_org'`.
    
        `'key_terms_in'` : list
            A boolean list repective to `'name'` indicating if at 
            least one key word from `'key_terms'` is present in the group name.

        `'num_groups'` : int
            The length of `result['name']`.
    """
    result = {
        'key_terms_in' : list(),
        'blacklist_in' : list(),
        'name' : list(),
    }

    for group_name in groups:
            
        if not other_org in group_name.lower():

            if list == type(key_terms):
                key_terms_in = any([k.lower() in group_name.lower() 
                                    for k in key_terms])
            else:
                key_terms_in = key_terms.lower() in group_name.lower()

            if list == type(blacklist):
                blacklist_in = any([bl.lower() in group_name.lower() 
                                    for bl in blacklist])
            elif str == type(blacklist):
                blacklist_in = blacklist.lower() in group_name.lower()
            else:
                blacklist_in = False
                
            result['key_terms_in'].append(key_terms_in)
            result['blacklist_in'].append(blacklist_in)
            result['name'].append(group_name)

    result['num_groups'] = len(result['name'])

    return result


def check_libs(libs, key_terms: str | list='', 
               blacklist: str | list | bool=False, other_org: str=''):
    """
    Checks libraries for desired `key_terms` in groups.

    Parameters
    ----------
    libs : dict
        A dictionary as `libs[library_name] = library_groups`.

    key_terms : str | list
        A `str` or `list` of `str`s to seach for in `libs` group names.

    other_org : str
        Only applicable when desired organism is `'human'` or 
        `'mouse'`. If desired organism is `'human'`, `other_org` 
        should be `'mouse'` and vice-versa.

    Returns
    -------
    summary, tally : pd.DataFrame | pd.DataFrame
        `summary` has cols `['Library', 'No. Gene Sets', 
        'No. Key Terms Matching']` where `'Library'` is the library from 
        `gseapy` with `'No. Gene Sets'` and `'No. Key Terms Matching'` 
        corresponding. `'No. Key Terms Matching'` only included if 
        `key_terms` argument is provided. `tally` has cols `['library', 
        'key_terms_in', 'name']` 
    """
    num_groups = dict()
    
    tally = {
        'library' : list(), 
        'key_terms_in' : list(), 
        'blacklist_in' : list(),
        'name' : list()
    }
    
    for library, groups in libs.items():
        res = check_groups(list(groups.keys()), key_terms, 
                           blacklist, other_org)

        lib_repeats = [library]*len(res['name'])

        tally['library'].extend(lib_repeats)
        tally['key_terms_in'].extend(res['key_terms_in'])
        tally['blacklist_in'].extend(res['blacklist_in'])
        tally['name'].extend(res['name'])

        num_groups[library] = res['num_groups']

    tally = pd.DataFrame(tally)

    key_dict = tally.copy()
    key_dict = key_dict.groupby('library')['key_terms_in'].sum().reset_index()
    key_dict = dict(zip(key_dict['library'], key_dict['key_terms_in']))

    bl_dict = tally.copy()
    bl_dict = tally.groupby('library')['blacklist_in'].sum().reset_index()
    bl_dict = dict(zip(bl_dict['library'], bl_dict['blacklist_in']))

    lib_names = np.unique(tally['library'])
    key_counts = [key_dict[l] for l in lib_names]
    bl_counts = [bl_dict[l] for l in lib_names]
    n_groups = [num_groups[l] for l in lib_names]

    summary = {
        'Library' : lib_names,
        'No. Gene Sets' : n_groups
    }

    if key_terms:
        summary['No. Key Terms Matching'] = key_counts
    if blacklist:
        summary['No. Blacklist Matching'] = bl_counts

    summary = pd.DataFrame(summary)

    return summary, tally


def find_candidates(organism: str='human', key_terms: str | list='', blacklist: str | list | bool=False):
    """
    Given `organism` and `key_terms`, will search for gene 
    groupings that could fit the datasets/classification task. 
    `blacklist` terms undesired in group names.

    Parameters
    ----------
    organism : str
        The species the gene grouping is for. Options are 
        `{'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'}`

    key_terms : str | list
        The types of cells or other specifiers the gene set is for 
        (example: 'CD4 T').

    blacklist : str | list | bool
        Term(s) undesired in group names. Ignored unless provided.

    Returns
    -------
    libraries : list
        A list of gene set library names that could serve for the 
        dataset/classification task.        

    Examples
    --------
    >>> scmkl.find_candidates('human', key_terms=' b ')
                                Library  No. Gene Sets
    0                    Azimuth_2023           1241
    1         Azimuth_Cell_Types_2021            341
    2   Cancer_Cell_Line_Encyclopedia            967
    3                 CellMarker_2024           1134
    No. Key Type Matching
    9
    9
    0
    21
    """
    check_organism(organism)

    if organism.lower() in global_lib_orgs:
        glo = global_lib_orgs.copy()
        glo.remove(organism)
        other_org = glo[0]
        libs = human_genesets
        libs = [name for name in libs if not other_org in name.lower()]
    else:
        libs = gp.get_library_name(organism)
        other_org = ''

    libs = {name : gp.get_library(name, organism)
            for name in libs}
    
    libs_df, _ = check_libs(libs, key_terms, blacklist, other_org)

    return libs_df


def get_gene_groupings(lib_name: str, organism: str='human', key_terms: str | list='', 
                       blacklist: str | list | bool=False, min_overlap: int=2,
                      genes: list | tuple | pd.Series | np.ndarray | set=[]):
    """
    Takes a gene set library name and filters to groups containing 
    element(s) in `key_terms`. If genes is provided, will 
    ensure that there are at least `min_overlap` number of genes in 
    each group. Resulting groups will meet all of the before-mentioned 
    criteria if `isin_logic` is `'and'` | `'or'`.

    Parameters
    ----------
    lib_name : str
        The desired library name.

    organism : str
        The species the gene grouping is for. Options are 
        `{'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'}`.

    key_terms : str | list
        The types of cells or other specifiers the gene set is for 
        (example: 'CD4 T').

    genes : array_like
        A vector of genes from the reference/query datasets. If not 
        assigned, function will not filter groups based on feature 
        overlap.

    min_overlap : int
        The minimum number of genes that must be present in a group 
        for it to be kept. If `genes` is not given, ignored.

    Returns
    -------
    lib : dict
        The filtered library as a `dict` where keys are group names 
        and keys are features.

    Examples
    --------
    >>> dataset_feats = [
    ...    'FUCA1', 'CLIC4', 'STMN1', 'SYF2', 'TAS1R1', 
    ...    'NOL9', 'TAS1R3', 'SLC2A5', 'THAP3', 'IGHM', 
    ...    'MARCKS', 'BANK1', 'TNFRSF13B', 'IGKC', 'IGHD', 
    ...    'LINC01857', 'CD24', 'CD37', 'IGHD', 'RALGPS2'
    ...    ]
    >>> rna_grouping = scmkl.get_gene_groupings(
    ...   'Azimuth_2023', key_terms=[' b ', 'b cell', 'b '], 
    ...   genes=dataset_feats)
    >>>
    >>> rna_groupings.keys()
    dict_keys(['PBMC-L1-B Cell', 'PBMC-L2-Intermediate B Cell', ...])
    """
    check_organism(organism)
    
    lib = gp.get_library(lib_name, organism)

    if organism.lower() in global_lib_orgs:
        glo = global_lib_orgs.copy()
        glo.remove(organism)
        other_org = glo[0]
    else:
        other_org = ''

    group_names = list(lib.keys())
    res = check_groups(group_names, key_terms, blacklist, other_org)
    del res['num_groups']

    # Finding groups where group name matches key_terms
    g_summary = pd.DataFrame(res)

    if key_terms:
        kept = g_summary['key_terms_in']
        kept_groups = g_summary['name'][kept].to_numpy()
        g_summary = g_summary[kept]
    else:
        print("Not filtering with `key_terms` parameter.")
        kept_groups = g_summary['name'].to_numpy()

    if blacklist:
        kept = ~g_summary['blacklist_in']
        kept_groups = g_summary['name'][kept].to_numpy()
    else:
        print("Not filtering with `blacklist` parameter.")
    
    # Filtering library
    lib = {group : lib[group] for group in kept_groups}

    if 0 < len(genes):
        del_groups = list()
        genes = list(set(genes.copy()))
        for group, features in lib.items():
            overlap = np.isin(features, genes)
            overlap = np.sum(overlap)
            if overlap < min_overlap:
                print(overlap, flush=True)
                del_groups.append(group)

        # Removing genes without enough overlap
        for group in del_groups:
            print(f'Removing {group} from grouping.')
            del lib[group]

    else:
        print("Not checking overlap between group and dataset features.")

    return lib
