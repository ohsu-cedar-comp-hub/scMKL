import numpy as np
import pandas as pd


def find_overlap(start1, end1, start2, end2) -> bool:
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


def get_atac_groupings(gene_library : dict, feature_names : list | np.ndarray | pd.Series, gene_annotations : pd.DataFrame) -> dict:
    '''
    Function to create an ATAC region grouping for scMKL using genes in a gene set library.
    Searches for regions in gene_annotations that overlap with assay features, then matches gene_names to genes in gene_library to create grouping.
    Input:
        gene_library : a dictionary with gene set names as keys and a set | list | np.ndarray of gene names
        feature_names : an array of feature regions in scATAC assay
        gene_annotations : a pd.DataFrame with columns [chr, start, stop, gene_name] where [chr, start, stop] for each row is the region of the gene_name gene body
    Output:
        ATAC_group_dict : a grouping dictionary with gene set names from gene_library as keys and an array of regions as values.
    '''
    assert ('chr' and 'start' and 'end' and 'gene_name') in gene_annotations.columns, "gene_annotations argument must be a dataframe with columns ['chr', 'start', 'end', 'gene_name']"

    # Variables for region comparison and grouping creation
    peak_gene_dict = {}
    ga_regions = {}
    feature_dict = {}
    ATAC_grouping = {group : [] for group in gene_library.keys()}

    # Creating a list of all gene names to filter gene annotations by, ensuring there are no NaN values in list
    all_genes = [gene for group in gene_library.keys() for gene in gene_library[group] if type(gene) != float]

    # Filtering gene_annotations by genes in the gene_library
    gene_annotations = gene_annotations[np.isin(gene_annotations['gene_name'], all_genes)]

    # Creating dictionaries from gene_annotations where:
        # peak_gene_dict - (chr, start_location, end_location) : gene_name
        # ga_regions - chr : np.ndarray([[start_location, end_location], [start_location, end_location], ...])
    for i, anno in gene_annotations.iterrows():
        peak_gene_dict[(anno['chr'], int(anno['start']), int(anno['end']))] = anno['gene_name']
        if anno['chr'] in ga_regions.keys():
            ga_regions[anno['chr']] = np.concatenate((ga_regions[anno['chr']], np.array([[anno['start'], anno['end']]], dtype = int)), axis = 0)
        else:
            ga_regions[anno['chr']] = np.array([[anno['start'], anno['end']]], dtype=int)

    print("Gene Annotations Formatted", flush = True)

    # Reformatting feature names to a list of lists where each element is a list of [chr, start_location, stop_location]
    feature_names = [peak.split("-") for peak in feature_names]
    # Creating a dictionary of features from assay where chr : np.ndarray([[start_location, end_location], [start_location, end_location], ...])
    for peak_set in feature_names:
        if peak_set[0] in feature_dict.keys():
            feature_dict[peak_set[0]] = np.concatenate((feature_dict[peak_set[0]], np.array([[peak_set[1], peak_set[2]]], dtype = int)), axis = 0)
        else:
            feature_dict[peak_set[0]] = np.array([[peak_set[1], peak_set[2]]], dtype = int)

    print("Assay Peaks Formatted", flush = True)

    # This is where the regions in the assay and the regions in the annotations are compared then genes are matched between the gene_library and gene_annotation for the respective regions
    # Iterating through all the chromosomes in the feature assay
    print("Comparing Regions", flush = True)
    for chrom in feature_dict.keys():
        # Continuing if chromosom for iteration not in gene_annotations to reduce number of comparisons
        if chrom not in ga_regions.keys():
            continue
        # Iterating through peaks in features for the given chromosome
        for region in feature_dict[chrom]:
            # Iteration through peaks in ga_regions (from gene_annotations) for the current chromosome during the iteration
            for anno in ga_regions[chrom]:
                # Checking if the current feature peak and ga_region peak overlap
                if find_overlap(region[0], region[1], anno[0], anno[1]):
                    gene = peak_gene_dict[(chrom, anno[0], anno[1])]
                    # Iterating through all of the gene sets in gene_library to match gene for current ga_annotation peak to genes in gene sets
                    for group in gene_library.keys():
                        if gene in gene_library[group]:
                            # Adding feature region to group in ATAC_grouping dict   
                            ATAC_grouping[group].append("-".join([chrom, str(region[0]), str(region[1])]))

        print(f'{chrom} Comparisons Complete', flush = True)

    # Returning a dictionary with keys from gene_library keys and values are arrays of peaks from feature array that overlap with gene peaks from gene_annotations if respective genes are in gene_library[gene_set]
    return ATAC_grouping


def create_results_df():
    '''
    '''
    pass