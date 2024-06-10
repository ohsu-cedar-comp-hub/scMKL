import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed
from keras import metrics
import scipy
import argparse
import time
import tracemalloc
import pickle
import os

import tensorflow as tf

tracemalloc.start()


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

seed = np.random.default_rng(100 * replication)
tf.random.set_seed(100 * replication)
set_random_seed(100 * replication)

tracemalloc.start()
seed_obj = np.random.default_rng(replication * 100)

datatype = 'counts' if assay in ['rna', 'gene_scores'] else 'binary'

X = scipy.sparse.load_npz(f'data/scMM/{dataset}/{dataset}_{assay.upper()}_scipy.npz')
cell_labels = np.load(f'data/scMM/{dataset}/{dataset}_cell_metadata.npy', allow_pickle= True)

y = np.zeros(cell_labels.shape)
y[cell_labels == np.unique(cell_labels)[1]] = 1

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

    col_vars = Sparse_Var(X[train_indices,:], axis = 0)
    mvf = np.argsort(col_vars).ravel()[-5000:]

    X = X[:, mvf]

    assay = f'{assay}_mvf'

X_train = X[train_indices,:].toarray()
X_test = X[test_indices,:].toarray()

y_train = y[train_indices]
y_test = y[test_indices]


batch_sizes = [64,128, 256, 512, 1024]
k = 4

# Cross Validation
positive_indices = np.where(y_train == np.unique(y_train)[0])[0]
negative_indices = np.setdiff1d(np.arange(len(y_train)), positive_indices)

positive_annotations = np.arange(len(positive_indices)) % k
negative_annotations = np.arange(len(negative_indices)) % k

auroc_array = np.zeros((k, len(batch_sizes)))

for fold in np.arange(k):
    fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
    fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))    

    fold_train_X, fold_test_X = Process_Data(X_train[fold_train,:], X_train[fold_test, :], data_type= datatype, return_dense= True)

    
    for n in np.arange(len(batch_sizes)):

        model = Sequential()
        for _ in np.arange(5):
            model.add(Dense(512, activation = 'relu'))
            model.add(Dropout(0.1))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.AUC()]) 
        callback = EarlyStopping(monitor = 'loss', patience = 3, start_from_epoch = 1)
        history = model.fit(fold_train_X, y_train[fold_train], batch_size= batch_sizes[n], epochs= 10, validation_split=0.0, callbacks = [callback], verbose = 2)

        score = model.evaluate(fold_test_X, y_train[fold_test], verbose=0)

        auroc_array[fold, n] = score[1]

opt_batch = batch_sizes[np.argmax(np.mean(auroc_array, axis = 0))]

X_train, X_test = Process_Data(X_train, X_test, datatype, return_dense= True)

model = Sequential()
for _ in np.arange(5):
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.1))

model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', metrics.AUC(), metrics.Precision(), metrics.Recall(), metrics.F1Score()]) 

callback = EarlyStopping(monitor = 'val_loss', patience = 3, start_from_epoch = 1)

start = time.time()
history = model.fit(X_train, y_train, batch_size= opt_batch, epochs= 100, validation_split=0.1, callbacks = [callback])

score = model.evaluate(X_test, y_test, verbose=0)
end = time.time()



metric_dict = {'Accuracy': score[1], # 1 - score because somewhere the labels were flipped
               'AUROC': score[2],
               'Precision': score[3],
               'Recall': score[4],
               'F1-Score': score[5][0]}

results = {}
results['Metrics'] = metric_dict
results['Inference_time'] = end - start
results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'
results['History'] = history # contains model loss and accuracy across all training epochs
results['Model'] = model

results_dir = f'results/scMM/{dataset}/MLP_results/result_files/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# with open(f'{results_dir}/{dataset}_{assay}_MLP_replication_{replication}.pkl', 'wb') as fout:
#     pickle.dump(results, fout)



