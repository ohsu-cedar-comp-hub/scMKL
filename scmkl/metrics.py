import numpy as np
import sklearn.metrics as skm


def _get_metric(metric, y, y_pred, y_test, probabilities, p_cl):
    '''
    Takes truth values and probabilities/predictions and returns the 
    given metric.
    '''
    metric_ops = ['AUROC', 'F1-Score', 'Accuracy', 'Recall', 'Precision']
    assert metric in metric_ops, f'Metric must be one of {metric_ops}'

    match metric:
        case 'AUROC':
            fpr, tpr, _ = skm.roc_curve(y, probabilities)
            value = skm.auc(fpr, tpr)
        case 'Accuracy':
            value = np.mean(y_test == y_pred)
        case 'F1-Score':
            value = skm.f1_score(y_test, y_pred, 
                                 pos_label = p_cl)
        case 'Precision':
            value = skm.precision_score(y_test, y_pred, 
                                        pos_label = p_cl)
        case 'Recall':
            value = skm.recall_score(y_test, y_pred, 
                                     pos_label = p_cl)
            
    return value


def calculate_metric(adata, metric)-> float:
    '''
    Function to calculate the metric for a classification. 
    Designed as a helper function for scmkl.optimize_alpha(). 
    Recommended to use scmkl.predict() for model evaluation.
    
    Parameters
    ----------  
    adata : adata object with trained model and Z matrices in 
            uns

    metric : string of ['AUROC', 'F1-Score', 'Accuracy', 'Recall', 
             'Precision'] to calculate metric for.

    Returns
    -------
    Calculated metric value
    '''
    # Capturing test indices
    train_idx = adata.uns['train_indices']

    # Defining positive class
    classes = np.unique(adata.obs['labels'].iloc[train_idx].to_numpy())
    p_cl = classes[0]

    y_test = adata.obs['labels'].iloc[train_idx].to_numpy()
    y_test = y_test.ravel()
    X_test = adata.uns['Z_test']

    assert X_test.shape[0] == len(y_test), (f"X has {X_test.shape[0]} "
                                            "samples and y has "
                                            f"{len(y_test)} samples.")

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-adata.uns['model'].predict(X_test)))
    
    # Group Lasso requires 'continous' y values need to re-descritize it
    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1

    #Convert numerical probabilities into binary phenotype
    y_pred = np.array(np.repeat(classes[1], X_test.shape[0]), 
                      dtype = 'object')
    y_pred[np.round(probabilities, 0).astype(int) == 1] = classes[0]
    
    value = _get_metric(metric, y, y_pred, y_test, probabilities, p_cl)
    
    return value

