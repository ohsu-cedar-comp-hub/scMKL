import src.scMKL_src as src
import numpy as np
import pickle

x = np.load('example/data/TCGA-ESCA.npy', allow_pickle = True)
labels = np.load('example/data/TCGA-ESCA_cell_metadata.npy', allow_pickle = True)
features = np.load('example/data/TCGA-ESCA_RNA_feature_names.npy', allow_pickle = True)


with open('example/data/RNA_hallmark_groupings.pkl', 'rb') as fin:
    group_dict = pickle.load(fin)

D = int(np.sqrt(len(labels)) * np.log(np.log(len(labels))))

x, features = src.Downselect_Features(x, features, group_dict)

sigmas = src.Calculate_Sigma(x, group_dict, 'rna', features)

train_indices, test_indices = src.Train_Test_Split(labels, seed_obj= np.random.RandomState(500))

x_train = x[train_indices,:]
x_test = x[test_indices,:]
y_train = labels[train_indices]
y_test = labels[test_indices]


sigmas = src.Optimize_Sigma(X = x_train, y = y_train, group_dict = group_dict, assay = 'rna', D = D, feature_set = features, sigma_list = sigmas)

Z_train, Z_test = src.Calculate_Z(x_train, x_test, group_dict, 'rna', D, features, sigmas)

gl = src.Train_Model(Z_train, y_train, 2 * D)
auc = src.Calculate_Auroc(gl, Z_test, y_test)
selected_groups = src.Find_Selected_Pathways(gl, group_names= group_dict.keys())