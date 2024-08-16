import numpy as np 
import pandas as pd
from plotnine import *
import src.scMKL_src_anndata as src
import re


def plot_group_norms(norm_dict, group_names, groups_to_display = 10):

    '''
    Function to plot group norms as they vary across alpha values.
    Input:
        norm_dict: Dictionary with alpha values as keys and numpy arrays of the norms as values
        group_names: numpy array of group names corresponding with the norms in the dictionary
        groups_to_display: Either an integer number of the top selected groups to display
                           Or iterable with desired group names to view
    Output:
        Line plot of norm vs alpha 
    '''

    group_names = np.array(group_names)

    norm_df = pd.DataFrame(np.zeros((len(group_names)*len(norm_dict), 3)))
    norm_df.columns = ['Norm', 'Alpha', 'Group Name']
    norm_df.Alpha = np.repeat(list(norm_dict.keys()), len(group_names))
    norm_df['Group Name'] = np.tile(group_names, len(norm_dict))

    for i, alpha in enumerate(norm_dict.keys()):
        alpha_indices = np.arange(i * len(group_names), (i+1) * len(group_names))

        norm_df.iloc[alpha_indices,0] = norm_dict[alpha]


    if isinstance(groups_to_display, int):
        rowsums = norm_df.groupby(['Group Name'], observed = False).sum('Norm')
        top_groups = np.array(rowsums.index)[np.argsort(rowsums.Norm.to_numpy().ravel())[-10:]]
        norm_df = norm_df.iloc[np.where(np.isin(norm_df['Group Name'], top_groups))[0], :]
    else:
        norm_df = norm_df.iloc[np.where(np.isin(norm_df['Group Name'], groups_to_display))[0], :]

    norm_plot = (ggplot(norm_df)
                    +  geom_line(aes(x = 'Alpha', y = 'Norm', color = 'Group Name'))
                    +  scale_x_continuous(breaks = np.unique(norm_df.Alpha))
                    +  theme(figure_size=(10,5))
                    +  theme_classic())
    return norm_plot
    
def plot_classification_metrics(metric_dict):

    '''
    Function to plot results of classification across alpha with multiple metrics. Assumes all the same metrics we calculated for each alpha
    Input:
        metric_dict: Dictionary with results.  
                Keys are the tested alpha values. Values are a nested dictionary with the metrics run as keys and that metric 'score' as values
    Output:
        Bar plot showing results with a facet for each metric
    '''

    for i, alpha in enumerate(metric_dict.keys()):

        if i == 0:
            num_alpha = len(metric_dict)
            num_metrics = len(metric_dict[alpha])
            plotting_df = pd.DataFrame(np.zeros((num_alpha * num_metrics, 3)))
            plotting_df.columns = ['Score','Metric', 'Alpha']

        alpha_indices = np.arange(i * num_metrics, (i+1) * num_metrics)
        plotting_df.iloc[alpha_indices, 2] = alpha
        plotting_df.iloc[alpha_indices,1] = list(metric_dict[alpha].keys())
        plotting_df.iloc[alpha_indices,0] = list(metric_dict[alpha].values())
        
    bar_plot = (ggplot(plotting_df)
        + geom_col(aes(x = 'Alpha', y = 'Score', fill = 'Metric'), position = 'position_dodge')
        + scale_x_continuous(breaks = list(metric_dict.keys()))
        + theme_classic())
    
    return bar_plot

def Feature_Set_Enrichment(X: np.ndarray, y: np.ndarray, feature_names: np.ndarray, feature_groupings: dict, data_type: str, metric = 'correlation'):

    if data_type.lower() == 'binary':
        X = src.TF_IDF_filter(X, mode = 'normalize')

    metric_scores = np.zeros(feature_names.shape)
    enrichment_scores = enrichment_scores = pd.DataFrame({'Pathway_name': feature_groupings.keys(), 'Enrichment': np.ones(len(feature_groupings.keys()))})

    num_y = np.ones(y.shape)
    num_y[y == np.unique(y)[1]] = -1

    for i in np.arange(len(feature_names)):

        if metric.lower() == 'correlation':
            metric_scores[i] = np.corrcoef(X[:,i].ravel(), num_y)[0,1]


    
    order = np.flip(np.argsort(metric_scores))

    feature_names = feature_names[order]
    metric_scores = metric_scores[order]

    for i, pathway in enumerate(feature_groupings.keys()):

 
        pathway_features = feature_groupings[pathway]

        overlapping_indices = np.where(np.in1d(feature_names, pathway_features, assume_unique= True))[0]
        pathway_corr_sum = np.sum(np.abs(metric_scores[overlapping_indices]))

        running_sum = 0
        ES = 0

        for j in range(len(feature_names)):
            if feature_names[j] in pathway_features:
                running_sum += np.abs(metric_scores[j]) / pathway_corr_sum

            else:
                running_sum -= 1 / (len(feature_names) - len(pathway_features))

            if np.abs(running_sum) > np.abs(ES):
                ES = running_sum



        enrichment_scores.iloc[i,1] = ES

    return enrichment_scores

def find_overlap(range1, range2):

    len_overlapping_bp = abs(min(range1[1], range2[1]) - max(range1[0], range2[0]))

    return len_overlapping_bp

def split_peaks(peak):
    return tuple(re.split(':|-', peak))

def Create_Groupings(feature_array: np.ndarray, original_groupings: pd.DataFrame, min_overlap = 0):

    '''
    Function to convert peaks from database bed files to the peaks in your dataset
    Input:
        feature_array- numpy array containing the peak names in the data.
        original_groupings- Dataframe with grouping information.  Should have 2 columns, all of the peaks and their corresponding feature group
    Output:
        group_dict- dictionary with group names from original_groupings as keys and their corresponding peaks from feature_array as values
    '''

    database_peaks = pd.DataFrame(np.zeros((original_groupings.shape[0], 4)))
    database_peaks.columns = ['Group', 'Chr', 'Start', 'Stop']
    database_peaks.loc[:,'Group'] = np.array(original_groupings.iloc[:,0])
    database_peaks.iloc[:,1:] = [split_peaks(peaks) for peaks in original_groupings.iloc[:,1]]
    
    database_peaks.Start = database_peaks.Start.astype(int)
    database_peaks.Stop = database_peaks.Stop.astype(int)

    group_dict = {}
    for group in np.unique(original_groupings['Group']):
        group_dict[group] = set()

    for i, peak in enumerate(feature_array):
        if i % 1000 == 0:
            print(i)
        chr, start, stop = split_peaks(peak)
        start = int(start)
        stop = int(stop)

        chr_peaks = database_peaks.iloc[np.where(database_peaks.Chr == chr)[0],:]

        chr_peaks = chr_peaks.iloc[np.where(np.logical_and(chr_peaks.Stop > start, chr_peaks.Start < stop))[0],:]

        chr_peaks['Overlap'] = list(map(lambda x,y: find_overlap((start, stop), (x,y)), chr_peaks.Start, chr_peaks.Stop)) 

        groups_w_peak = np.unique(chr_peaks['Group'].iloc[np.where(chr_peaks.Overlap > min_overlap)[0]])

        for group in groups_w_peak:
            group_dict[group].add(peak)

    return group_dict

def convert_dict_to_df(group_dict):
    
    
    grouping_df = pd.DataFrame(np.zeros((0, 2)))
    grouping_df.columns = ['Group', 'Peak']

    for group, peaks in group_dict.items():

        group_df = pd.DataFrame({'Group': np.repeat(group, len(peaks)).ravel(),
                                    'Peak': np.array(list(peaks))})
        
        grouping_df = pd.concat((grouping_df, group_df))


    return grouping_df

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

def get_ATAC_groupings(gene_library : dict, feature_names : list | np.ndarray | pd.Series, gene_annotations : pd.DataFrame) -> dict:
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
    assert ('chr' and 'start' and 'end' and 'gene_name') in gene_annotations.columns, "gene_annotations argument must have column names ['chr', 'start', 'end', 'gene_name']"

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