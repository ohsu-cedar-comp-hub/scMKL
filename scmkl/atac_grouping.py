import pandas as pd
import numpy as np
import re


def _find_overlap(start1 : int, end1 : int, start2 : int, end2 : int) -> bool:
    '''
    Function to determine whether two regions on the same chromosome overlap.
    Input:
        start1 : the start position for region 1
        end1 : the end position for region 1
        start2 : the start position for region 2
        end2: the end postion for region 2
    Output:
        True if the regions overlap by 1bp
        False if the regions do not overlap
    '''
    return max(start1,start2) <= min(end1, end2)


def _get_tss(row : pd.DataFrame) -> int:
    '''
    Takes a row from a DataFrame as a DataFrame with columns 
    ['start', 'end', 'strand'] and returns the transcription start site
    depending on gene strandedness.
    Args:
        row - a pd.DataFrame from a gtf as a pd.DataFrame containing
            only one row and columns ['start', 'end', 'strand'].
    Returns:
        The transcription start site for row's respective annotation.
    '''
    if row.iloc[2] == '+':
        return row.iloc[0]
    
    elif row.iloc[2] == '-':
        return row.iloc[1]
    
    else:
        raise ValueError('Strand symbol must be "+" or "-"')
    

def _calc_range(row : pd.DataFrame, len_up : int, len_down : int) -> list:
    '''
    Returns an infered promotor region for given annotation range 
    depending on transcription start site and user-defined upstream and
    downstream adjustments.
    Args:
        row - a pd.DataFrame from a gtf as a pd.DataFrame containing
            only one row and columns ['tss', 'strand'] where 'tss' is
            the transcription start site.
        len_up - how many base pairs upstream from the transcription 
            start site the promotor range should be adjusted for.
        len_down - how many base pairs downstream from the 
            transcription start site the promotor range should be 
            adjusted for.
    Returns:
        A 2 element list where the first element is the adjusted start
        position and the second the the adjusted end position.
    '''
    if row.iloc[1] == '+':
        start = row.iloc[0] - len_up
        end = row.iloc[0] + len_down

    elif row.iloc[1] == '-':
        start = row.iloc[0] - len_down
        end = row.iloc[0] + len_up

    else:
        raise ValueError('Strand symbol must be "+" or "-"')
    
    return [start, end]


def _adjust_regions(gene_anno : pd.DataFrame, len_up : int, len_down : int
                    ) -> pd.DataFrame:
    '''
    Takes a GTF file as a pd.DataFrame and adjusts start and end 
    positions to represent promotor regions given user-defined 
    upstream and downstream adjustments relative to the transcription
    start site.
    Args:
        gene_anno - a pd.DataFrame with columns ['chr', 'start', 'end', 
            'strand', 'gene_name'] created from a GTF file.
        len_up - an int for how many base pairs upstream of the 
            transcription start site the promotor region should be 
            adjusted for.
        len_down - an int for how many base pairs downstream of the 
            transcription start site the promotor region should be 
            adjusted for.
    Returns:
        gene_anno as a pd.DataFrame where ['start', 'end'] columns 
        represent the start and end positions of inferred promotor
        regions for each annotation.
    '''
    # Subsetting DataFrame to only required data
    required_cols = ['chr', 'start', 'end', 'strand', 'gene_name']
    gene_anno = gene_anno[required_cols]

    # Adding a column that annotates transcription start site based on strand
    tss = gene_anno[['start', 'end', 'strand']].apply(_get_tss, axis = 1)
    gene_anno.insert(1, column = 'tss', value = tss)

    # Calculating start and end positions based on the tss
    adj_regions = gene_anno[['tss', 'strand']].apply(lambda row:
                                                    _calc_range(
                                                        row = row,
                                                        len_up = len_up,
                                                        len_down = len_down
                                                        ), 
                                                   axis = 1, 
                                                   result_type = 'expand'
                                                   )
    
    gene_anno.loc[:, ['start', 'end']] = adj_regions.values
    
    return gene_anno


def _create_region_dicts(gene_anno : pd.DataFrame) -> dict:
    '''
    Takes a GTF as a pd.DataFrame and returns dictionaries required for
    region comparisons between gene_annotations and assay features.
    Args:
        gene_anno - a pd.DataFrame with columns ['chr', 'start', 'end', 
            'strand', 'gene_name'] created from a GTF file.
    Returns:
        Two dictionaries. 
            peak_gene_dict is a dictionary where keys are regions and
            values are genes.
            ga_regions is a dictionary where chromosomes are keys and
            regions are values.
    '''
    peak_gene_dict = {}
    ga_regions = {}
    for i, anno in gene_anno.iterrows():
        peak_gene_dict[(anno['chr'], int(anno['start']), int(anno['end']))] = anno['gene_name']
        if anno['chr'] in ga_regions.keys():
            ga_regions[anno['chr']] = np.concatenate((ga_regions[anno['chr']], np.array([[anno['start'], anno['end']]], dtype = int)), axis = 0)
        else:
            ga_regions[anno['chr']] = np.array([[anno['start'], anno['end']]], dtype=int)

    return peak_gene_dict, ga_regions


def _create_feature_dict(feature_names : list | set | np.ndarray | pd.Series
                         ) -> dict:
    '''
    Takes an array of feature names and returns data formatted for
    feature comparisons.
    Args:
        feature_names - an iterable object of region names from 
        single-cell ATAC data matrix.
    Returns:
        A dictionary where keys are chromosomes and values are regions.
    '''
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


def _compare_regions(feature_dict : dict, ga_regions : dict,
                     peak_gene_dict : dict, gene_sets : dict, chr_sep : str
                     ) -> dict:
    '''
    Takes features from a single-cell ATAC data matrix and regions from
    a gene annotation file to return an ATAC grouping where regions 
    from feature_dict and regions from gene annotations overlap.
    Args:
        feature_dict - a dictionary where keys are chromosomes and 
            values are regions. This data should come from a single-
            cell atac experiment.
        ga_regions - a dictionary where keys are chromosomes and values
            are regions. This data should come from a gene annotations 
            file.
        peak_gene_dict - a dictionary where keys are peaks from gene 
            annotation file and values are the gene they are associated
            with.
        gene_sets - a dictionary where keys are gene set names and 
            values are an iterable object of gene names.
        chr_sep - The character that separates the chromosome from the 
            rest of the region in the original feature array.
    Returns:
        A dictionary where keys are the names from gene_sets and values
        are a list of regions from feature_names that overlap with 
        promotor regions respective to genes in gene sets (i.e., if 
        ATAC feature in feature_names overlaps with promotor region 
        from a gene in a gene set from gene_sets, that region will be
        added to the new dictionary under the respective gene set 
        name).
    '''
    atac_grouping = {group : [] for group in gene_sets.keys()}

    for chrom in feature_dict.keys():
        if chrom not in ga_regions.keys():
            continue
        for region in feature_dict[chrom]:
            for anno in ga_regions[chrom]:
                if _find_overlap(region[0], region[1], anno[0], anno[1]):
                    gene = peak_gene_dict[(chrom, anno[0], anno[1])]
                    for group in gene_sets.keys():
                        if gene in gene_sets[group]:
                            atac_grouping[group].append(chrom + chr_sep + str(region[0]) + "-" + str(region[1]))

    return atac_grouping


def get_atac_groupings(gene_anno : pd.DataFrame, gene_sets : dict, 
                       feature_names : np.ndarray | pd.Series | list | set,
                       len_up : int = 5000, len_down : int = 5000) -> dict:
    '''
    Takes processed gene annotations in GTF format, and gene set 
    library, and feature names from a single-cell ATAC data matrix
    and creates an ATAC region set to be used with scMKL.
    Args:
        gene_anno - gene annotations in GTF format as a pd.DataFrame 
            with columns ['chr', 'start', 'end', 'strand', 'gene_name']
        gene_sets - a dictionary with gene_set names as keys and an 
            iterable object of gene_names as values.
        feature_names - an iterable, one dimension object containing 
            feature names corresponding to a single_cell ATAC data 
            matrix.
    Returns:
        A dictionary where keys are the names from gene_sets and values
        are a list of regions from feature_names that overlap with 
        promotor regions respective to genes in gene sets (i.e., if 
        ATAC feature in feature_names overlaps with promotor region 
        from a gene in a gene set from gene_sets, that region will be
        added to the new dictionary under the respective gene set 
        name).
    '''
    # Getting a unique set of gene names from gene_sets
    all_genes = {gene for group in gene_sets.keys()
                 for gene in gene_sets[group]}
    
    # Filtering out NaN values
    all_genes = [gene for gene in all_genes if type(gene) != float]

    # Filtering out annotations for genes not present in gene_sets
    gene_anno = gene_anno[np.isin(gene_anno['gene_name'], all_genes)]

    # Adjusting start and end columns to represent promotor regions
    gene_anno = _adjust_regions(gene_anno = gene_anno, 
                                len_up = len_up, len_down = len_down)

    # Creating a dictionary from assay features where [chr] : (start, end)
    feature_dict = _create_feature_dict(feature_names)

    # Creating data structures from gene_anno for comparing regions
    peak_gene_dict, ga_regions = _create_region_dicts(gene_anno)

    # Capturing the separator type used in assay
    chr_sep = '-' if '-' in feature_names[0] else ':'

    atac_groupings = _compare_regions(feature_dict = feature_dict,
                                     ga_regions = ga_regions,
                                     peak_gene_dict = peak_gene_dict,
                                     gene_sets = gene_sets,
                                     chr_sep = chr_sep)
    
    return atac_groupings