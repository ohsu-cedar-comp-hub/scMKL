import numpy as np 
import pandas as pd
from plotnine import *

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