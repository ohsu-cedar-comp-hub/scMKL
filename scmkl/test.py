import numpy as np
import sklearn



def predict(adata, metrics = None):
    '''
    Function to return predicted labels and calculate any of AUROC, Accuracy, F1 Score, Precision, Recall for a classification. 
    Input:  
            adata- adata object with trained model and Z matrices in uns
            metrics- Which metrics to calculate on the predicted values

    Output:
            Values predicted by the model
            Dictionary containing AUROC, Accuracy, F1 Score, Precision, and/or Recall depending on metrics argument

    '''
    y_test = adata.obs['labels'].iloc[adata.uns['test_indices']].to_numpy()
    X_test = adata.uns['Z_test']
    assert X_test.shape[0] == len(y_test), 'X and y must have the same number of samples'
    assert all([metric in ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall'] for metric in metrics]), 'Unknown metric provided.  Must be one or more of AUROC, Accuracy, F1-Score, Precision, Recall'

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-adata.uns['model'].predict(X_test)))

    # Group Lasso requires 'continous' y values need to re-descritize it
    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1

    metric_dict = {}

    #Convert numerical probabilities into binary phenotype
    y_pred = np.array(np.repeat(np.unique(y_test)[1], len(y_test)), dtype = 'object')
    y_pred[np.round(probabilities,0).astype(int) == 1] = np.unique(y_test)[0]

    if metrics == None:
        return y_pred

    if 'AUROC' in metrics:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y, probabilities)
        metric_dict['AUROC'] = sklearn.metrics.auc(fpr, tpr)
    if 'Accuracy' in metrics:
        metric_dict['Accuracy'] = np.mean(y_test == y_pred)
    if 'F1-Score' in metrics:
        metric_dict['F1-Score'] = sklearn.metrics.f1_score(y_test, y_pred, pos_label = np.unique(y_test)[0])
    if 'Precision' in metrics:
        metric_dict['Precision'] = sklearn.metrics.precision_score(y_test, y_pred, pos_label = np.unique(y_test)[0])
    if 'Recall' in metrics:
        metric_dict['Recall'] = sklearn.metrics.recall_score(y_test, y_pred, pos_label = np.unique(y_test)[0])

    return y_pred, metric_dict


def find_selected_groups(adata) -> np.ndarray:

    '''
    Function to find feature groups selected by the model during training.  If feature weight assigned by the model is non-0, then the group containing that feature is selected.
    Inputs:
        model- A trained celer.GroupLasso model.
        group_names- An iterable object containing the group_names in the same order as the feature groupings from Data array.
    Outpus:
        Numpy array containing the names of the groups selected by the model.
    '''

    selected_groups = []
    coefficients = adata.uns['model'].coef_
    group_size = adata.uns['model'].get_params()['groups']
    group_names = np.array(list(adata.uns['group_dict'].keys()))


    for i, group in enumerate(group_names):
        if not isinstance(group_size, (list, set, np.ndarray, tuple)):
            group_norm = np.linalg.norm(coefficients[np.arange(i * group_size, (i+1) * group_size - 1)])
        else: 
            group_norm = np.linalg.norm(coefficients[group_size[i]])

        if group_norm != 0:
            selected_groups.append(group)

    return np.array(selected_groups)


def calculate_auroc(adata)-> float:
    '''
    Function to calculate the AUROC for a classification. 
    Designed as a helper function.  Recommended to use Predict() for model evaluation.
    Input:  
            adata- adata object with trained model and Z matrices in uns
    Output:
            Calculated AUROC value
    '''

    y_test = adata.obs['labels'].iloc[adata.uns['test_indices']].to_numpy()
    X_test = adata.uns['Z_test']

    y_test = y_test.ravel()
    assert X_test.shape[0] == len(y_test), f'X has {X_test.shape[0]} samples and y has {len(y_test)} samples.'

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-adata.uns['model'].predict(X_test)))
    # Group Lasso requires 'continous' y values need to re-descritize it

    y = np.zeros((len(y_test)))
    y[y_test == np.unique(y_test)[0]] = 1
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, probabilities)
    auc = sklearn.metrics.auc(fpr, tpr)
    
    return(auc)


def extract_kernel_weights():
    '''
    '''
    pass