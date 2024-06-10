import numpy as np
import scipy
import argparse
import time
import tracemalloc
import pickle
import os
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

tracemalloc.start()


def Train_Test_Split(y, seed_obj = np.random.default_rng(100), train_ratio = 0.8):
    '''
    Function to calculate training and testing indices for given dataset, preserving the ratio of each class in y.
    Input:
            y- Numpy array of cell labels. Can have any number of classes for this function.
            seed_obj- Numpy random state used for random processes. Can be specified for reproducubility or set by default.
            train_ratio- decimal value ratio of features in training:testing sets
    Output:
            train_indices- Array of indices of training cells
            test_indices- Array of indices of testing cells
    '''

    unique_labels = np.unique(y)
    train_indices = []

    for label in unique_labels:

        # Find index of each unique label
        label_indices = np.where(y == label)[0]

        # Sample these indices according to train ratio
        train_label_indices = seed_obj.choice(label_indices, int(len(label_indices) * train_ratio), replace = False)
        train_indices.extend(train_label_indices)

    # Test indices are the indices not in the train_indices
    test_indices = np.setdiff1d(np.arange(len(y)), train_indices, assume_unique = True)

    return train_indices, test_indices

def Process_Data(X_train, X_test, data_type = 'counts', return_dense = True):

    '''
    Function to preprocess data matrix according to type of data (counts- e.g. rna, or binary- atac)
    Will process test data according to parameters calculated from test data

    Input:
        X_train- A scipy sparse or numpy array
        X_train- A scipy sparse or numpy array
        data_type- 'counts' or 'binary'.  Determines what preprocessing is applied to the data. 
            Log transforms and standard scales counts data
            TFIDF filters ATAC data to remove uninformative columns
    Output:
        X_train, X_test- Numpy arrays with the process train/test data respectively.
    '''

    # Remove features that have no variance in the training data (will be uniformative)
    assert data_type in ['counts', 'binary'], 'Improper value given for data_type'

    var = Sparse_Var(X_train, axis = 0)
    variable_features = np.where(var > 1e-5)[0]

    X_train = X_train[:,variable_features]
    X_test = X_test[:, variable_features]

    #Data processing according to data type
    if data_type.lower() == 'counts':

        if scipy.sparse.issparse(X_train):
            X_train = X_train.log1p()
            X_test = X_test.log1p()
        else:
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)
            
        #Center and scale count data
        train_means = np.mean(X_train, 0)
        train_sds = np.sqrt(var[variable_features])

        X_train = (X_train - train_means) / train_sds
        X_test = (X_test - train_means) / train_sds
    
    elif data_type.lower() == 'binary':

        # TFIDF filter binary peaks
        non_empty_row = np.where(np.sum(X_train, axis = 1) > 0)[0]

        if scipy.sparse.issparse(X_train):
            non_0_cols = TF_IDF_filter(X_train.toarray()[non_empty_row,:], mode= 'filter')
        else:
            non_0_cols = TF_IDF_filter(X_train[non_empty_row,:], mode = 'filter')

        X_train = X_train[:, non_0_cols]
        X_test = X_test[:, non_0_cols]

    if return_dense and scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    return X_train, X_test

def TF_IDF_filter(X, mode = 'filter'):
    '''
    Function to use Term Frequency Inverse Document Frequency filtering for atac data to find meaningful features. 
    If input is pandas data frame or scipy sparse array, it will be converted to a numpy array.
    Input:
            x- Data matrix of cell x feature.  Must be a Numpy array or Scipy sparse array.
            mode- Argument to determine what to return.  Must be filter or normalize
    Output:
            TFIDF- Output depends on given 'mode' parameter
                'filter'- returns which column sums are non 0 i.e. which features are significant
                'normalize'- returns TFIDF filtered data matrix of the same dimensions as x. Returns as scipy sparse matrix
    '''

    assert mode in ['filter', 'normalize'], 'mode must be "filter" or "normalize".'
    
    if scipy.sparse.issparse(X):
        row_sum = np.array(X.sum(axis=1)).flatten()
        tf = scipy.sparse.csc_array(X / row_sum[:, np.newaxis])
        doc_freq = np.array(np.sum(X > 0, axis=0)).flatten()
    else:
        row_sum = np.sum(X, axis=1, keepdims=True)
        tf = X / row_sum    
        doc_freq = np.sum(X > 0, axis=0)

    idf = np.log1p((1 + X.shape[0]) / (1 + doc_freq))
    tfidf = tf * idf

    if mode == 'normalize':
        if scipy.sparse.issparse(tfidf):
            tfidf = scipy.sparse.csc_matrix(tfidf)
        return tfidf
    elif mode == 'filter':
        significant_features = np.where(np.sum(tfidf, axis=0) > 0)[0]
        return significant_features

def Sparse_Var(X, axis = None):

    '''
    Function to calculate variance on a sparse matrix.
    Input:
        X- A scipy sparse or numpy array
        axis- Determines which axis variance is calculated on. Same usage as Numpy
            axis = 0 => column variances
            axis = 1 => row variances
            axis = None => total variance (calculated on all data)
    Output:
        var- Variance values calculated over the given axis
    '''

    # E[X^2] - E[X]^2
    if scipy.sparse.issparse(X):
        var = np.array((X.power(2).mean(axis = axis)) - np.square(X.mean(axis = axis)))
    else:
        var = np.var(X, axis = axis)
    return var.ravel()

def Filter_Features(X, feature_set, group_dict):
    '''
    Function to remove unused features from X matrix.  Any features not included in group_dict will be removed from the matrix.
    Also puts the features in the same relative order (of included features)
    Input:
            X- Data array. Can be Numpy array or Scipy Sparse Array
            feature_names- Numpy array of corresponding feature names
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
    Output:
            X- Data array containing data only for features in the group_dict
            feature_names- Numpy array of corresponding feature names from group_dict
    '''
    assert X.shape[1] == len(feature_set), 'Given features do not correspond with features in X'    

    group_features = set()

    # Store all objects in dictionary in array
    for group in group_dict.keys():
        group_features.update(set(group_dict[group]))

    # Find location of desired features in whole feature set
    group_feature_indices = np.nonzero(np.in1d(feature_set, np.array(list(group_features)), assume_unique = True))[0]

    # Subset only the desired features and their data
    X = X[:,group_feature_indices]
    feature_set = feature_set[group_feature_indices]

    return X, feature_set

def Predict(model, X_test, y_test, metrics = None):
    '''
    Function to return predicted labels and calculate any of AUROC, Accuracy, F1 Score, Precision, Recall for a classification. 
    Input:  
            Trained model- Must be compatible with sklearn based ML model
            X_test- the Matrix containing the testing data (will be Z_test in this workflow)
            y_test- Corresponding labels for testing data.  Needs to binary, but not necessarily numeric
            metrics- Which metrics to calculate on the predicted values

    Output:
            Values predicted by the model
            Dictionary containing AUROC, Accuracy, F1 Score, Precision, and/or Recall depending on metrics argument

    '''
    y_test = y_test.ravel()
    assert X_test.shape[0] == len(y_test), 'X and y must have the same number of samples'
    assert all([metric in ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall'] for metric in metrics]), 'Unknown metric provided.  Must be one or more of AUROC, Accuracy, F1-Score, Precision, Recall'

    # Sigmoid function to force probabilities into [0,1]
    probabilities = 1 / (1 + np.exp(-model.predict(X_test)))

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

parser = argparse.ArgumentParser(description='Run classification for unimodal data with MLP based on MAPS paper')
parser.add_argument('-r' ,'--replication', help = 'Which replication to run', type = int, default = 1)
parser.add_argument('-a', '--assay', help = 'Which assay to run with', type = str)
parser.add_argument('-d', '--dataset', help = 'Which dataset to use', type = str)
parser.add_argument('-s', '--feature_subset', help = 'Which_features to subset the data fram', choices = [None, 'hallmark', 'mvf'])
args = parser.parse_args()
dataset = args.dataset
assay = args.assay
replication = args.replication
subset = args.feature_subset

seed = np.random.default_rng(100 * replication)

tracemalloc.start()
seed_obj = np.random.default_rng(replication * 100)

datatype = 'counts' if assay in ['rna', 'gene_scores'] else 'binary'

X = scipy.sparse.load_npz(f'data/scMM/{dataset}/{dataset}_{assay.upper()}_scipy.npz')
cell_labels = np.load(f'data/scMM/{dataset}/{dataset}_cell_metadata.npy', allow_pickle= True)

y = np.ones(cell_labels.shape)
y[cell_labels == np.unique(cell_labels)[1]] = 0

train_indices, test_indices = Train_Test_Split(y, seed_obj= np.random.default_rng(100 * replication))

if subset == 'hallmark':
    feature_names = np.load(f'data/scMM/{dataset}/{dataset}_{assay.upper()}_feature_names.npy', allow_pickle= True)
    with open(f'data/scMM/{dataset}/{dataset}_groupings.pkl', 'rb') as fin:
        groupings = pickle.load(fin)

    group_dict = {}
    for pathway in groupings['hallmark'].keys():
        group_dict[pathway] = groupings['hallmark'][pathway]['genes' if assay in ['rna', 'gene_scores'] else 'peaks']

    X, _ = Filter_Features(X, feature_names, group_dict)

    assay = f'{assay}_hallmark'
elif subset == 'mvf':

    col_vars = Sparse_Var(X, axis = 0)
    mvf = np.argsort(col_vars).ravel()[-5000:]

    X = X[:, mvf]

    assay = f'{assay}_mvf'


X_train = X[train_indices,:].toarray()
X_test = X[test_indices,:].toarray()

y_train = y[train_indices]
y_test = y[test_indices]

model  = HistGradientBoostingClassifier()

start = time.time()
print('Starting Cross Validation')

max_depths = [1,10,20, 30,40]
k = 4
auroc_array = np.zeros((k, len(max_depths)))

positive_indices = np.where(y_train == np.unique(y_train)[0])[0]
negative_indices = np.setdiff1d(np.arange(len(y_train)), positive_indices)

positive_annotations = np.arange(len(positive_indices)) % k
negative_annotations = np.arange(len(negative_indices)) % k

for fold in np.arange(k):
    fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
    fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))    

    fold_train_X, fold_test_X = Process_Data(X_train[fold_train,:], X_train[fold_test,:], data_type= datatype, return_dense= True)

    for i, depth in enumerate(max_depths):
        model = HistGradientBoostingClassifier(max_depth= depth).fit(fold_train_X, y_train[fold_train])

        _, auroc = Predict(model = model, X_test= fold_test_X, y_test= cell_labels[train_indices][fold_test].ravel(), metrics= ['AUROC'])

        auroc_array[fold, i] = auroc['AUROC']
        
opt_depth = max_depths[np.argmax(np.mean(auroc_array, axis = 0))]

X_train, X_test = Process_Data(X_train, X_test, datatype, return_dense= True)

model = HistGradientBoostingClassifier(max_depth= opt_depth).fit(X_train, y_train)

predictions, metrics = Predict(model, X_test, cell_labels[test_indices].ravel(), metrics = ['AUROC','F1-Score', 'Accuracy', 'Precision', 'Recall'])
end = time.time()

results = {}
results['Metrics'] = metrics
results['Predictions'] = predictions
results['Inference_time'] = end - start
results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'
results['Model'] = model
results['Depth_star'] = opt_depth

results_dir = f'results/scMM/{dataset}/XGBoost_results/result_files/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# with open(f'{results_dir}/{dataset}_{assay}_XGBoost_replication_{replication}.pkl', 'wb') as fout:
#     pickle.dump(results, fout)



