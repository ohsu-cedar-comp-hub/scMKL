import numpy as np 
import pandas as pd
from plotnine import *
import scMKL_src as src
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
