import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import muon
from muon import atac as ac
import itertools
from math import ceil
from sklearn import metrics
from plotnine import (ggplot, aes, theme_classic, ylim, element_text, theme,
                      geom_point, scale_x_reverse, annotate, geom_bar, 
                      coord_flip, element_blank, labs, geom_tile, 
                      scale_fill_gradient, facet_wrap, 
                      scale_color_manual, scale_color_gradient)

from scmkl.dataframes import (_parse_result_type, get_weights, sort_groups, 
                              format_group_names)


def _get_alpha(alpha: None | float, result: dict, is_multiclass: bool):
    """
    Gets the smallest alpha from a results file. Works for both binary 
    and multiclass results.
    """
    if type(alpha) == float:
        return alpha
    
    if 'Alpha_star' in result.keys():
        return result['Alpha_star']
    
    if is_multiclass:
        classes = list(result['Classes'])
        alpha_list = list(result[classes[0]]['Norms'].keys())
        alpha = np.min(alpha_list)

    else:
        alpha_list = list(result['Norms'].keys())
        alpha = np.min(alpha_list)

    return alpha


def color_alpha_star(alphas, alpha_star, color):
    """
    Takes an array of alphas and returns a list of the same size where 
    each element is `'black'` except where 
    `alpha_star == alphas`, which will be `'gold'`.

    Parameters
    ----------
    alphas : list | tuple | np.ndarray
        The 1D array of alphas.

    alpha_star: float
        The best performing alpha from cross-validation.

    color : str
        The color of all alphas other than `alpha_star`.

    Returns
    -------
    c_array, c_dict : np.ndarray, dict
        `c_array` is the array of colors corresponding to alphas. 
        `c_dict` is the color dict with alphas as keys and color as 
        values.
    """
    c_array = np.array([color] * len(alphas), dtype='<U15')
    as_pos = np.where(alphas == alpha_star)[0]
    c_array[as_pos] = 'gold'

    c_dict = {alphas[i] : c_array[i]
              for i in range(len(alphas))}

    return c_array, c_dict



def plot_conf_mat(results, title = '', cmap = None, normalize = True,
                          alpha = None, save = None) -> None:
    """
    Creates a confusion matrix from the output of scMKL.

    Parameters
    ----------
    results : dict
        The output from either scmkl.run() or scmkl.one_v_rest()
        containing results from scMKL.

    title : str
        The text to display at the top of the matrix.

    cmap : matplotlib.colors.LinearSegmentedColormap
        The gradient of the values displayed from `matplotlib.pyplot`.
        If `None`, `'Purples'` is used see matplotlib color map 
        reference for more information. 

    normalize : bool
        If `False`, plot the raw numbers. If `True`, plot the 
        proportions.

    alpha : None | float
        Alpha that matrix should be created for. If `results` is from
        `scmkl.one_v_all()`, this is ignored. If `None`, smallest alpha
        will be used.

    save : None | str
        File path to save plot. If `None`, plot is not saved.

    Returns
    -------
    None
    
    Examples
    --------
    >>> # Running scmkl and capturing results
    >>> results = scmkl.run(adata = adata, alpha_list = alpha_list)
    >>> 
    >>> from matplotlib.pyplot import get_cmap
    >>> 
    >>> scmkl.plot_conf_mat(results, title = '', cmap = get_cmap('Blues'))

    ![conf_mat](../tests/figures/plot_conf_mat_binary.png)

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html
    """
    # Determining type of results
    if ('Observed' in results.keys()) and ('Metrics' in results.keys()):
        multi_class = False
        names = np.unique(results['Observed'])
    else:
        multi_class = True
        names = np.unique(results['Truth_labels'])

    if multi_class:
        cm = metrics.confusion_matrix(y_true = results['Truth_labels'], 
                              y_pred = results['Predicted_class'], 
                              labels = names)
    else:
        min_alpha = np.min(list(results['Metrics'].keys()))
        alpha = alpha if alpha != None else min_alpha
        cm = metrics.confusion_matrix(y_true = results['Observed'],
                              y_pred = results['Predictions'][alpha],
                              labels = names)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Purples')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    acc_label = 'Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'
    acc_label = acc_label.format(accuracy, misclass)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(acc_label)

    if save != None:
        plt.savefig(save)
        plt.clf()
    else:
        plt.show()

    return None


def plot_metric(summary_df : pd.DataFrame, alpha_star = None, 
                x_axis: str='Alpha', color = 'red'):
    """
    Takes a data frame of model metrics and optionally alpha star and
    creates a scatter plot given metrics against alpha values. For 
    multiclass results, `alpha_star` is not shown and points are 
    colored by class.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Dataframe created by `scmkl.get_summary()`.

    alpha_star : None | float
        If `not None`, a label will be added for tuned `alpha_star` 
        being optimal model parameter for performance from cross 
        validation on the training data. Can be calculated with 
        `scmkl.optimize_alpha()`. Is ignored if `summary_df` is from a 
        multiclass result.

    x_axis : str
        Must be either `'Alpha'` or `'Number of Selected Groups'`. Is 
        the variable that will be plotted on the x-axis.

    color : str
        Color to make points on plot.

    Returns
    -------
    metric_plot : plotnine.ggplot.ggplot
        A plot with alpha values on x-axis and metric on y-axis.

    Examples
    --------
    >>> results = scmkl.run(adata, alpha_list)
    >>> summary_df = scmkl.get_summary(results)
    >>> metric_plot = plot_metric(results)
    >>>
    >>> metric_plot.save('scMKL_performance.png')

    ![metric_plot](../tests/figures/plot_metric_binary.png)
    """
    # Capturing metric from summary_df
    metric_options = ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
    metric = np.intersect1d(metric_options, summary_df.columns)[0]

    x_axis_ticks = np.unique(summary_df[x_axis])
    summary_df['Alpha Star'] = summary_df['Alpha'] == alpha_star

    if 'Class' in summary_df.columns:
        color_lab = 'Class'
    else:
        c_dict = {
            True : 'gold',
            False : color
            }
        color_lab = 'Alpha Star'

    if np.min(summary_df[metric]) < 0.5:
        min_y = 0
    else:
        min_y = 0.6

    plot = (ggplot(summary_df, aes(x = x_axis, y = metric, color=color_lab)) 
            + geom_point(size=4)
            + theme_classic()
            + ylim(min_y, 1)
            + scale_x_reverse(breaks=x_axis_ticks)
            + theme(
                axis_text=element_text(weight='bold', size=10),
                axis_title=element_text(weight='bold', size=12),
                legend_title=element_text(weight='bold', size=14),
                legend_text=element_text(weight='bold', size=12)
            )
            )
    
    if not 'Class' in  summary_df:
        plot += scale_color_manual(c_dict)
        
    return plot


def weights_barplot(result, n_groups: int=1, alpha: None | float=None, 
                 color: str='red'):
    """
    Plots the top `n_groups` weighted groups for each cell class. Works 
    for a single scmkl result (either multiclass or binary).

    Parameters
    ----------
    result : dict
        The output of `scmkl.run()`.

    n_groups : int
        The number of top groups to plot for each cell class.

    alpha : None | float
        The alpha parameter to create figure for. If `None`, the 
        smallest alpha is used.

    color : str
        The color the bars should be.

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        A barplot of weights.

    Examples
    --------
    >>> result = scmkl.run(adata, alpha_list)
    >>> plot = scmkl.plot_weights(result)

    ![weights_barplot](../tests/figures/weights_barplot_binary.png)
    """
    is_multi, is_many = _parse_result_type(result)
    assert not is_many, "This function only supports single results"

    alpha = _get_alpha(alpha, result, is_multi)
    df = get_weights(result)

    # Subsetting to only alpha and filtering to top groups
    df = df[df['Alpha'] == alpha]
    if is_multi:
        g_order = set(df['Group'])
        for ct in result['Classes']:
            cor_ct = df['Class'] == ct
            other_ct = ~cor_ct

            temp_df = df.copy()
            temp_df = temp_df[cor_ct]
            temp_df = temp_df.sort_values('Kernel Weight', ascending=False)
            top_groups = temp_df['Group'].iloc[0:n_groups].to_numpy()
            
            cor_groups = np.isin(df['Group'], top_groups)
            filter_array = np.logical_and(cor_ct, cor_groups)

            filter_array = np.logical_or(filter_array, other_ct)

            df = df[filter_array]

        df['Group'] = format_group_names(df['Group'], rm_words = ['Markers'])
            

    else:
        df = df.sort_values('Kernel Weight', ascending=False)
        df = df.iloc[0:n_groups]
        df['Group'] = format_group_names(df['Group'], rm_words = ['Markers']) 
        g_order = sort_groups(df)[::-1]
        df['Group'] = pd.Categorical(df['Group'], g_order)

    plot = (ggplot(df)
            + theme_classic()
            + coord_flip()
            + labs(y=f'Kernel Weight (λ = {alpha})')
            + theme(
                axis_text=element_text(weight='bold', size=10),
                axis_title=element_text(weight='bold', size=12),
                axis_title_y=element_blank()
            )
            )

    # This needs to be reworked for multiclass runs
    if is_multi:
        height = (3*ceil((len(set(df['Class'])) / 3)))
        print(height)
        plot += geom_bar(aes(x='Group', y='Kernel Weight'), 
                         stat='identity', fill=color)
        plot += facet_wrap('Class', scales='free', ncol=3)
        plot += theme(figure_size=(15,height))
    else:
        plot += geom_bar(aes(x='Group', y='Kernel Weight'), 
                         stat='identity', fill=color)
        plot += theme(figure_size=(7, 9))

    return plot


def weights_heatmap(result, n_groups: None | int=None, 
                    class_lab: str | None=None, low: str='white', 
                    high: str='red', alpha: float | None=None,
                    scale_weights: bool=False):
    """
    Plots a heatmap of kernel weights with groups on the y-axis and 
    alpha on the x-axis if binary result. If a multiclass result, one 
    alpha is used per class and the x-axis is class.

    Parameters
    ----------
    result : dict
        The output of `scmkl.run()`.

    n_groups : int
        The number of top groups to plot. Not recommended for 
        multiclass results.

    class_lab : str | None
        For multiclass results, if `not None`, will only plot group 
        weights for `class_lab`.

    low : str
        The color for low kernel weight.

    high : str
        The color for high kernel weight.

    alpha : None | float
        The alpha parameter to create figure for. If `None`, the 
        smallest alpha is used.

    scale_weights : bool
        If `True`, the the kernel weights will be scaled for each group 
        within each class. Ignored if result is from a binary 
        classification.

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        A heatmap of weights.

    Examples
    --------
    >>> result = scmkl.run(adata, alpha_list)
    >>> plot = scmkl.plot_weights(result)

    ![weights_heatmap](../tests/figures/weights_heatmap_binary.png)
    """
    is_multi, is_many = _parse_result_type(result)
    assert not is_many, "This function only supports single results"

    if type(class_lab) is str:
        result = result[class_lab]

    df = get_weights(result)
    df['Group'] = format_group_names(df['Group'], ['Markers'])

    # Filtering and sorting values
    sum_df = df.groupby('Group')['Kernel Weight'].sum()
    sum_df = sum_df.reset_index()
    order = sort_groups(sum_df)[::-1]
    df['Group'] = pd.Categorical(df['Group'], categories=order)

    if type(n_groups) is int:
        sum_df = sum_df.sort_values(by='Kernel Weight', ascending=False)
        top_groups = sum_df.iloc[0:n_groups]['Group'].to_numpy()
        df = df[np.isin(df['Group'], top_groups)]
    else:
        n_groups = len(set(df['Group']))

    df['Alpha'] = pd.Categorical(df['Alpha'], np.unique(df['Alpha']))

    if n_groups > 40:
        fig_size = (7,8)
    elif n_groups < 25:
        fig_size = (7,6)
    else: 
        fig_size = (7,8)

    if 'Class' in df.columns:
        alpha = _get_alpha(alpha, result, is_multi)
        df = df[df['Alpha'] == alpha]
        x_lab = 'Class'
    else:
        x_lab = 'Alpha'

    if scale_weights and is_multi:
        max_norms = dict()
        for ct in set(df['Class']):
            g_rows = df['Class'] == ct
            max_norms[ct] = np.max(df[g_rows]['Kernel Weight'])
            scale_cols = ['Class', 'Kernel Weight']

        new = df[scale_cols].apply(lambda x: x[1] / max_norms[x[0]], axis=1)
        df['Kernel Weight'] = new

        l_title = 'Scaled\nKernel Weight'

    else:
        l_title = 'Kernel Weight'

    plot = (ggplot(df, aes(x=x_lab, y='Group', fill='Kernel Weight'))
            + geom_tile(color='black')
            + scale_fill_gradient(high=high, low=low)
            + theme_classic()
            + theme(
                figure_size=fig_size,
                axis_text=element_text(weight='bold', size=10),
                axis_text_x=element_text(rotation=90),
                axis_title=element_text(weight='bold', size=12),
                axis_title_y=element_blank(),
                legend_title=element_text(text=l_title, weight='bold', size=12),
                legend_text=element_text(weight='bold', size=10)
            ))

    return plot


def weights_dotplot(result, n_groups: None | int=None, 
                    class_lab: str | None=None, low: str='white', 
                    high: str='red', alpha: float | None=None, 
                    scale_weights: bool=False):
    """
    Plots a dotplot of kernel weights with groups on the y-axis and 
    alpha on the x-axis if binary result. If a multiclass result, one 
    alpha is used per class and the x-axis is class.

    Parameters
    ----------
    result : dict
        The output of `scmkl.run()`.

    n_groups : int
        The number of top groups to plot. Not recommended for 
        multiclass results.

    class_lab : str | None
        For multiclass results, if `not None`, will only plot group 
        weights for `class_lab`.

    low : str
        The color for low kernel weight.

    high : str
        The color for high kernel weight.

    alpha : None | float
        The alpha parameter to create figure for. If `None`, the 
        smallest alpha is used.

    scale_weights : bool
        If `True`, the the kernel weights will be scaled for each 
        within each class.

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        A barplot of weights.

    Examples
    --------
    >>> result = scmkl.run(adata, alpha_list)
    >>> plot = scmkl.plot_weights(result)

    ![weights_dotplot](../tests/figures/weights_dotplot_binary.png)
    """
    is_multi, is_many = _parse_result_type(result)
    assert not is_many, "This function only supports single results"

    if type(class_lab) is str:
        result = result[class_lab]

    df = get_weights(result)
    df['Group'] = format_group_names(df['Group'], ['Markers'])

    # Filtering and sorting values
    sum_df = df.groupby('Group')['Kernel Weight'].sum()
    sum_df = sum_df.reset_index()
    order = sort_groups(sum_df)[::-1]
    df['Group'] = pd.Categorical(df['Group'], categories=order)

    if type(n_groups) is int:
        sum_df = sum_df.sort_values(by='Kernel Weight', ascending=False)
        top_groups = sum_df.iloc[0:n_groups]['Group'].to_numpy()
        df = df[np.isin(df['Group'], top_groups)]
    else:
        n_groups = len(set(df['Group']))

    df['Alpha'] = pd.Categorical(df['Alpha'], np.unique(df['Alpha']))

    if n_groups > 40:
        fig_size = (7,8)
    elif n_groups < 25:
        fig_size = (7,6)
    else: 
        fig_size = (7,8)

    if 'Class' in df.columns:
        alpha = _get_alpha(alpha, result, is_multi)
        df = df[df['Alpha'] == alpha]
        x_lab = 'Class'
    else:
        x_lab = 'Alpha'

    if scale_weights:
        max_norms = dict()
        for ct in set(df['Class']):
            g_rows = df['Class'] == ct
            max_norms[ct] = np.max(df[g_rows]['Kernel Weight'])
            scale_cols = ['Class', 'Kernel Weight']

        new = df[scale_cols].apply(lambda x: x[1] / max_norms[x[0]], axis=1)
        df['Kernel Weight'] = new

        l_title = 'Scaled\nKernel Weight'

    else:
        l_title = 'Kernel Weight'


    plot = (ggplot(df, aes(x=x_lab, y='Group', fill='Kernel Weight', color='Kernel Weight'))
            + geom_point(size=5)
            + scale_fill_gradient(high=high, low=low)
            + scale_color_gradient(high=high, low=low)
            + theme_classic()
            + theme(
                figure_size=fig_size,
                axis_text=element_text(weight='bold', size=10),
                axis_text_x=element_text(rotation=90),
                axis_title=element_text(weight='bold', size=12),
                axis_title_y=element_blank(),
                legend_title=element_text(text=l_title, weight='bold', size=12),
                legend_text=element_text(weight='bold', size=10)
            ))

    return plot


def group_umap(adata: ad.AnnData, g_name: str | list, is_binary: bool=False, 
               labels: None | np.ndarray | list=None, title: str='', 
               save: str=''):
    """
    Uses a scmkl formatted `ad.AnnData` object to show sample 
    separation using scmkl discovered groupings.

    Parameters
    ----------
    adata : ad.AnnData
        A scmkl formatted `ad.AnnData` object with `'group_dict'` in 
        `.uns`.

    g_name : str | list
        The groups who's features should be used to filter `adata`. If 
        is a list, features from multiple groups will be used.
    
    is_binary : bool
        If `True`, data will be processed using `muon` which includes 
        TF-IDF normalization and LSI.

    labels : None | np.ndarray | list
        If `None`, labels in `adata.obs['labels']` will be used to 
        color umap points. Else, provided labels will be used to color 
        points.

    title : str
        The title of the plot.

    save : str
        If provided, plot will be saved using `scanpy`'s `save` 
        argument. Should be the desired file name. Output will be 
        `<cwd>/figures/<save>`.

    Returns
    -------
    None

    Examples
    --------
    >>> adata_fp = 'data/_pbmc_rna.h5ad'
    >>> group_fp = 'data/_RNA_azimuth_pbmc_groupings.pkl'
    >>> adata = scmkl.format_adata(adata_fp, 'celltypes', group_fp, 
    ...                            allow_multiclass=True)
    >>> scmkl.group_umap(adata, 'CD16+ Monocyte Markers')

    ![group_umap](../tests/figures/umap_group_rna.png)
    """
    if list == type(g_name):
        feats = {feature 
                 for group in g_name 
                 for feature in adata.uns['group_dict'][group]}
        feats = np.array(list(feats))
    else:
        feats = np.array(list(adata.uns['group_dict'][g_name]))

    if labels:
        assert len(labels) == adata.shape[0], "`labels` do not match `adata`"
        adata.obs['labels'] = labels

    var_names = adata.var_names.to_numpy()

    col_filter = np.isin(var_names, feats)
    adata = adata[:, col_filter].copy()

    if not is_binary:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata, random_state=1)

    else:
        ac.pp.tfidf(adata, scale_factor=1e4)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        ac.tl.lsi(adata)
        sc.pp.scale(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
        sc.tl.umap(adata, random_state=1)

    if save:
        sc.pl.umap(adata, title=title, color='labels', save=save, show=False)

    else:
        sc.pl.umap(adata, title=title, color='labels')

    return None
