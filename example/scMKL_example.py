import numpy as np
import pickle

import sys

sys.path.insert(0, './src') # path to src file 
import scMKL_src as src

seed = np.random.default_rng(1)

X = np.load('example/data/TCGA-ESCA.npy', allow_pickle = True)
labels = np.load('example/data/TCGA-ESCA_cell_metadata.npy', allow_pickle = True)
features = np.load('example/data/TCGA-ESCA_RNA_feature_names.npy', allow_pickle = True)

with open('example/data/RNA_hallmark_groupings.pkl', 'rb') as fin:
    group_dict = pickle.load(fin)

D = int(np.sqrt(len(labels)) * np.log(np.log(len(labels))))

X, features = src.Filter_Features(X, features, group_dict)

train_indices, test_indices = src.Train_Test_Split(labels, seed_obj = seed)

X_train = X[train_indices,:]
X_test = X[test_indices,:]
y_train = labels[train_indices]
y_test = labels[test_indices]

sigmas = src.Estimate_Sigma(X = X_train, group_dict = group_dict, assay = 'rna', feature_set = features, seed_obj = seed)

sigmas = src.Optimize_Sigma(X = X_train, y = y_train, group_dict = group_dict, assay = 'rna', D = D, feature_set = features, 
                            sigma_list = sigmas, k = 2, sigma_adjustments = np.arange(0.1,2,0.1), seed_obj = seed)

Z_train, Z_test = src.Calculate_Z(X_train = X_train, X_test = X_test, group_dict = group_dict, assay = 'rna', D = D, 
                                  feature_set = features, sigma_list = sigmas, seed_obj = seed)

gl = src.Train_Model(Z_train, y_train, 2 * D, alpha = 1.1)
predictions, metrics = src.Predict(gl, Z_test, y_test, metrics = ['AUROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall'])
selected_groups = src.Find_Selected_Pathways(gl, group_names= list(group_dict.keys()))

print(metrics)
print(selected_groups)
