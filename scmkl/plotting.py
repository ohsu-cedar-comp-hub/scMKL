import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from plotnine import (ggplot, aes, theme_classic, ylim, 
                      geom_point, scale_x_reverse, annotate)


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
    plt.show()

    if save != None:
        plt.savefig(save)

    return None


def plot_metric(summary_df : pd.DataFrame, alpha_star = None, color = 'red'):
    """
    Takes a data frame of model metrics and optionally alpha star and
    creates a scatter plot given metrics against alpha values.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Dataframe created by `scmkl.get_summary()`.

    alpha_star : None | float
        > If `not None`, a label will be added for tuned `alpha_star` 
        being optimal model parameter for performance from cross 
        validation on the training data. Can be calculated with 
        `scmkl.optimize_alpha()`. 

    color : str
        Color to make points on plot.

    Returns
    -------
    metric_plot : plotnine.ggplot
        A plot with alpha values on x-axis and metric on y-axis.

    Examples
    --------
    >>> results = scmkl.run(adata, alpha_list)
    >>> summary_df = scmkl.get_summary(results)
    >>> metric_plot = plot_metric(results)
    >>>
    >>> metric_plot.save('scMKL_performance.png')
    """
    # Capturing metric from summary_df
    metric_options = ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
    metric = np.intersect1d(metric_options, summary_df.columns)[0]

    alpha_list = np.unique(summary_df['Alpha'])

    # Calculating alpha_star y_pos if present
    if alpha_star != None:
        best_rows = summary_df['Alpha'] == alpha_star
        alpha_star_metric = float(summary_df[best_rows][metric])

        metric_plot = (ggplot(summary_df, aes(x = 'Alpha', y = metric)) 
                        + geom_point(fill = color, color = color) 
                        + theme_classic() 
                        + ylim(0.6, 1)
                        + scale_x_reverse(breaks = alpha_list)
                        + annotate('text', x = alpha_star, 
                                   y = alpha_star_metric - 0.04, 
                                   label='|\nAlpha\nStar')
                        )
        
    else:
        metric_plot = (ggplot(summary_df, aes(x = 'Alpha', y = metric)) 
                + geom_point(fill = color, color = color) 
                + theme_classic() 
                + ylim(0.6, 1)
                + scale_x_reverse(breaks = alpha_list))
        
    return metric_plot