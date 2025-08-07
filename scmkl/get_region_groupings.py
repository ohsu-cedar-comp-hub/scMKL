import pandas as pd
import numpy as np
import re


def _find_overlap(start1 : int, end1 : int, start2 : int, end2 : int) -> bool:
    """
    Function to determine whether two regions on the same chromosome 
    overlap.
    
    Parameters
    ----------
    start1 : int
        The start position for region 1
    
    end1 : int
        The end position for region 1
    
    start2 : int
        The start position for region 2
    
    end2: int
        The end postion for region 2
    
    Returns
    -------
    is_overlapping : bool
        `True` if the regions overlap by 1bp. `False` if the regions 
        do not overlap
    """
    return max(start1, start2) <= min(end1, end2)


def get_tss(row : pd.DataFrame) -> int:
    """
    Takes a row from a DataFrame as a DataFrame with columns 
    ['start', 'end', 'strand'] and returns the transcription start site
    depending on gene strandedness.
    
    Parameters
    ----------
    row : pd.DataFrame 
        A gtf dataframe containing only one row and columns 
        `['start', 'end', 'strand']`.
    
    Returns
    -------
    tss : int
        The transcription start site for row's respective annotation.
    """
    if row.iloc[2] == '+':
        return row.iloc[0]
    
    elif row.iloc[2] == '-':
        return row.iloc[1]
    
    else:
        raise ValueError("Strand symbol must be '+' or '-'")
    

def calc_range(row : pd.DataFrame, len_up : int, len_down : int) -> list:
    """
    Returns an infered promotor region for given annotation range 
    depending on transcription start site and user-defined upstream 
    and downstream adjustments.

    Parameters
    ----------
    row : pd.DataFrame 
        A gtf as a dataframe containing only one row and columns 
        `['tss', 'strand']` where `'tss'` is the transcription start 
        site.
    
    len_up : int 
        Number of base pairs upstream from the transcription 
        start site the promotor range should be adjusted for.

    len_down : int
        Number of base pairs downstream from the transcription start 
        site the promotor range should be adjusted for.
    
    Returns
    -------
    start_end : list
        A 2 element list where the first element is the adjusted start
        position and the second the the adjusted end position.
    """
    if row.iloc[1] == '+':
        start = row.iloc[0] - len_up
        end = row.iloc[0] + len_down

    elif row.iloc[1] == '-':
        start = row.iloc[0] - len_down
        end = row.iloc[0] + len_up

    else:
        raise ValueError('Strand symbol must be "+" or "-"')
    
    return [start, end]


def adjust_regions(gene_anno : pd.DataFrame, len_up : int, len_down : int
                    ) -> pd.DataFrame:
    """
    Takes a GTF file as a pd.DataFrame and adjusts start and end 
    positions to represent promotor regions given user-defined 
    upstream and downstream adjustments relative to the transcription
    start site.

    Parameters
    ----------
    gene_anno : pd.DataFrame 
        A dataframe with columns `['chr', 'start', 'end', 'strand', 
        'gene_name']` created from a GTF file.
    
    len_up : int
        Number of base pairs upstream of the transcription start site 
        the promotor region should be adjusted for.
    
    len_down : int
        Number base pairs downstream of the transcription start site 
        the promotor region should be adjusted for.
    
    Returns
    -------
    gene_anno : pd.DataFrame
        A dataframe where `['start', 'end']` columns represent the 
        start and end positions of inferred promotor regions for each 
        annotation.
    """
    # Subsetting DataFrame to only required data
    required_cols = ['chr', 'start', 'end', 'strand', 'gene_name']
    gene_anno = gene_anno[required_cols]

    # Adding a column that annotates transcription start site based on strand
    tss = gene_anno[['start', 'end', 'strand']].apply(get_tss, axis = 1)
    gene_anno.insert(1, column = 'tss', value = tss)

    # Calculating start and end positions based on the tss
    adj_regions = gene_anno[['tss', 'strand']].apply(lambda row:
                                                     calc_range(
                                                        row = row,
                                                        len_up = len_up,
                                                        len_down = len_down), 
                                                   axis = 1, 
                                                   result_type = 'expand'
                                                   )
    
    gene_anno.loc[:, ['start', 'end']] = adj_regions.values
    
    return gene_anno


def create_region_dicts(gene_anno : pd.DataFrame) -> dict:
    """
    Takes a GTF as a pd.DataFrame and returns dictionaries required for
    region comparisons between gene_annotations and assay features.

    Parameters
    ----------
    gene_anno : pd.DataFrame 
        A dataframe with columns `['chr', 'start', 'end', 'strand', 
        'gene_name']` created from a GTF file.
    
    Returns
    -------
    peak_gene_dict : dict
        Keys are regions and values are genes.
        
    ga_regions : dict
        Chromosomes are keys and regions are values.
    """
    peak_gene_dict = {}
    ga_regions = {}

    for _, anno in gene_anno.iterrows():

        cur_chr = anno['chr']
        cur_start = int(anno['start'])
        cur_end = int(anno['end'])
        cur_name = anno['gene_name']

        peak_gene_dict[(cur_chr, cur_start, cur_end)] = cur_name

        cur_region = np.array([[cur_start, cur_end]], dtype = int)
        if cur_chr in ga_regions.keys():
            ga_regions[cur_chr] = np.concatenate((ga_regions[cur_chr], 
                                                  cur_region), axis = 0)
        
        else:
            ga_regions[cur_chr] = cur_region

    return peak_gene_dict, ga_regions


def create_feature_dict(feature_names : list | set | np.ndarray | pd.Series
                         ) -> dict:
    """
    Takes an array of feature names and returns data formatted for
    feature comparisons.
    
    Returns
    -------
    feature_names : array_like
        Array of region names from single-cell epigenetic data matrix.
    
    Returns
    -------
    feature_dict : dict
        Keys are chromosomes and values are regions.
    """
    feature_dict = {}
    feature_names = [re.split(":|-", peak) for peak in feature_names]
    for peak_set in feature_names:
        if peak_set[0] in feature_dict.keys():
            feature_dict[peak_set[0]] = np.concatenate(
                                                (feature_dict[peak_set[0]],
                                                    np.array([[peak_set[1], 
                                                                peak_set[2]]], 
                                                                dtype = int)), 
                                                                axis = 0)
        else:
            feature_dict[peak_set[0]] = np.array([[peak_set[1], 
                                                   peak_set[2]]], 
                                                   dtype = int)

    return feature_dict


def compare_regions(feature_dict : dict, ga_regions : dict,
                     peak_gene_dict : dict, gene_sets : dict, chr_sep : str
                     ) -> dict:
    """
    Takes features from a single-cell data matrix and regions from
    a gene annotation file to return an region grouping where regions 
    from feature_dict and regions from gene annotations overlap.

    Parameters
    ----------
    feature_dict : dict
        Keys are chromosomes and values are regions. This data should 
        come from a single-cell experiment.

    ga_regions : dict
        Keys are chromosomes and values are regions. This data should 
        come from a gene annotations (gtf) file.

    peak_gene_dict : dict
        Keys are peaks from gene annotation file and values are the 
        gene they are associated with.

    gene_sets : dict
        Keys are gene set names and values are an iterable object of 
        gene names.

    chr_sep : str
        The character that separates the chromosome from the rest of 
        the region in the original feature array.
    
    Returns
    -------
    epi_grouping : dict
        Keys are the names from gene_sets and values are a list of 
        regions from `feature_names` that overlap with promotor regions 
        respective to genes in `gene_sets` (i.e., if region in 
        `feature_names` overlaps with promotor region from a gene in a 
        gene set from `gene_sets`, that region will be added to the new 
        dictionary under the respective gene set name).
    """
    epi_grouping = {group : [] for group in gene_sets.keys()}

    for chrom in feature_dict.keys():
        if chrom not in ga_regions.keys():
            continue
        for region in feature_dict[chrom]:
            for anno in ga_regions[chrom]:
                if _find_overlap(region[0], region[1], anno[0], anno[1]):
                    gene = peak_gene_dict[(chrom, anno[0], anno[1])]
                    for group in gene_sets.keys():
                        if gene in gene_sets[group]:
                            feat_region = ''.join((chrom, chr_sep, 
                                                  str(region[0]), "-", 
                                                  str(region[1])))
                            epi_grouping[group].append(feat_region)

    return epi_grouping


def get_region_groupings(gene_anno : pd.DataFrame, gene_sets : dict, 
                       feature_names : np.ndarray | pd.Series | list | set,
                       len_up : int = 5000, len_down : int = 5000) -> dict:
    """
    Creates a peak set where keys are gene set names from `gene_sets` 
    and values are arrays of features pulled from `feature_names`. 
    Features are added to each peak set given overlap between regions 
    in single-cell data matrix and inferred gene promoter regions in 
    `gene_anno`.

    Parameters
    ----------
    gene_anno : pd.DataFrame
        Gene annotations in GTF format as a pd.DataFrame with columns
        `['chr', 'start', 'end', 'strand', 'gene_name']`.

    gene_sets : dict
        Gene set names as keys and an iterable object of gene names
        as values.

    feature_names : array_like | set
        Feature names corresponding to a single_cell epigenetic data 
        matrix.

    Returns
    -------
    epi_grouping : dict
        Keys are the names from `gene_sets` and values
        are a list of regions from `feature_names` that overlap with 
        promotor regions respective to genes in `gene_sets` (i.e., if 
        region in `feature_names` overlaps with promotor region from a 
        gene in a gene set from `gene_sets`, that region will be added 
        to the new dictionary under the respective gene set name).

    Examples
    --------
    >>> # Reading in a gene set and the peak names from dataset
    >>> gene_sets = np.load("data/RNA_hallmark_groupings.pkl", 
    ...                     allow_pickle = True)
    >>> peaks = np.load("data/MCF7_region_names.npy", 
    ...                 allow_pickle = True)
    >>> 
    >>> # Reading in GTF file
    >>> gtf_path = "data/hg38_subset_protein_coding.annotation.gtf"
    >>> gtf = scmkl.read_gtf(gtf_path, filter_to_coding=True)
    >>>
    >>> region_grouping = scmkl.get_region_groupings(gene_anno = gtf,
    ...                                         gene_sets = gene_sets,
    ...                                         feature_names = peaks)
    >>>
    >>> region_grouping.keys()
    dict_keys(['HALLMARK_TNFA_SIGNALING_VIA_NFKB', ...])
    """
    # Getting a unique set of gene names from gene_sets
    all_genes = {gene for group in gene_sets.keys()
                 for gene in gene_sets[group]}
    
    # Filtering out NaN values
    all_genes = [gene for gene in all_genes if type(gene) != float]

    # Filtering out annotations for genes not present in gene_sets
    gene_anno = gene_anno[np.isin(gene_anno['gene_name'], all_genes)]

    # Adjusting start and end columns to represent promotor regions
    gene_anno = adjust_regions(gene_anno = gene_anno, 
                                len_up = len_up, len_down = len_down)

    # Creating a dictionary from assay features where [chr] : (start, end)
    feature_dict = create_feature_dict(feature_names)

    # Creating data structures from gene_anno for comparing regions
    peak_gene_dict, ga_regions = create_region_dicts(gene_anno)

    # Capturing the separator type used in assay
    chr_sep = ':' if ':' in feature_names[0] else '-'

    epi_groupings = compare_regions(feature_dict = feature_dict,
                                     ga_regions = ga_regions,
                                     peak_gene_dict = peak_gene_dict,
                                     gene_sets = gene_sets,
                                     chr_sep = chr_sep)
    
    return epi_groupings
