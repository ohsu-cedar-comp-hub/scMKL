import numpy as np
import pickle

import sys

sys.path.insert(0, '/mnt/c/Users/kupp/Documents/scMKL_pkg/scMKL/src')
import scMKL_src as src

x = np.load('example/data/TCGA-ESCA.npy', allow_pickle = True)
labels = np.load('example/data/TCGA-ESCA_cell_metadata.npy', allow_pickle = True)
features = np.load('example/data/TCGA-ESCA_RNA_feature_names.npy', allow_pickle = True)

seed = np.random.RandomState(np.random.choice(np.arange(2000), 1))

with open('example/data/RNA_hallmark_groupings.pkl', 'rb') as fin:
    group_dict = pickle.load(fin)

D = int(np.sqrt(len(labels)) * np.log(np.log(len(labels))))

x, features = src.Filter_Features(x, features, group_dict)

sigmas = src.Calculate_Sigma(x, group_dict, 'rna', features, seed_obj = seed)

train_indices, test_indices = src.Train_Test_Split(labels, seed_obj = seed)

x_train = x[train_indices,:]
x_test = x[test_indices,:]
y_train = labels[train_indices]
y_test = labels[test_indices]


sigmas = src.Optimize_Sigma(X = x_train, y = y_train, group_dict = group_dict, assay = 'rna', D = D, feature_set = features, 
                            sigma_list = sigmas, k = 2, sigma_adjustments = np.arange(0.1,2,0.1), seed_obj = seed)

Z_train, Z_test = src.Calculate_Z(x_train, x_test, group_dict, 'rna', D, features, sigmas, seed_obj = seed)

gl = src.Train_Model(Z_train, y_train, 2 * D, alpha = 1.1)
auc = src.Calculate_Auroc(gl, Z_test, y_test)
selected_groups = src.Find_Selected_Pathways(gl, group_names= list(group_dict.keys()))

print(auc)
print(selected_groups)