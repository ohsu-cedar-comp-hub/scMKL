import pickle
import numpy as np
import gseapy as gp


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


def check_libs(libs, tissue, key_types):
    """
    Checks libraries for desired `tissues` or `key_types`.
    """
    tally = dict()
    for library, groups in libs.items():
        tally[library] = {'tissue_in' : list(), 'key_types_in' : list()}
        for group_name in groups:
            if list == type(tissue):
                tissue_in = any([t.lower() in group_name.lower() 
                                 for t in tissue])
            else:
                tissue_in = tissue in group_name.lower()

            if list == type(key_types):
                key_types_in = any([k.lower() in group_name.lower() 
                                    for k in key_types])
            else:
                key_types_in = key_types.lower() in group_name.lower()

            tally[library]['tissue_in'].append(tissue_in)
            tally[library]['key_types_in'].append(key_types_in)




def find_candidates(organism, tissue: str | list='', key_types: str | list=''):
    """
    Given `species`, `tissue`, and `key_types`, will search for gene 
    groupings that could fit the datasets/classification task.

    Parameters
    ----------
    species : str
        The species the gene grouping is for. Options are 
        `{'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'}`

    tissue : str | list
        The tissue the gene set is for. If `None`, will be ignored.

    key_types : str | list
        The types of cells or other specifiers the gene set is for 
        (example: 'CD4 T').

    Returns
    -------
    libraries : list
        A list of gene set library names that could serve for the 
        dataset/classification task.        

    Examples
    --------
    >>>
    """
    org_opts = {'human', 'mouse', 'yeast', 'fly', 'fish', 'worm'}
    org_err = f"Invalid `organism`, choose from {org_opts}"
    assert organism.lower() in org_opts, org_err

    global_lib_orgs = ['human', 'mouse']

    if organism.lower() in global_lib_orgs:
        global_lib_orgs.remove(organism)
        other_org = global_lib_orgs[0]
        libs = human_genesets
        libs = [name for name in libs if not other_org in name.lower()]
    else:
        libs = gp.get_library_name(organism)

    if tissue or key_types:
        check_libs()

    # libs = {name : gp.get_library(name, organism)
    #         for name in libs}
    
    return libs




def get_gene_grouping(species, tissue, key_types):
    """
    
    """
    pass

print(find_candidates('human'))