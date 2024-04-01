import numpy as np
from scipy.sparse import load_npz
import pickle
import time
import tracemalloc
import sys

sys.path.insert(0, '..')
import src.scMKL_src as src


tracemalloc.start()

# Option to change replication for different train/test splits to assess model stability
replication = 4

seed = np.random.default_rng(100*replication)

rna_X = load_npz('data/MCF7_RNA_X.npz')
rna_feature_names = np.load(f'data/MCF7_RNA_feature_names.npy', allow_pickle= True)

atac_X = load_npz(f'data/MCF7_ATAC_X.npz')
atac_feature_names = np.load('data/MCF7_ATAC_feature_names.npy', allow_pickle= True)

cell_labels = np.load(f'data/MCF7_cell_labels.npy', allow_pickle= True)

with open(f'data/MCF7_feature_groupings.pkl', 'rb') as fin:
    feature_groupings = pickle.load(fin)

rna_group_dict = {}
atac_group_dict = {}

for pathway in feature_groupings.keys():
    rna_group_dict[pathway] = feature_groupings[pathway]['genes']
    atac_group_dict[pathway] = feature_groupings[pathway]['peaks']

rna_X, rna_feature_names = src.Filter_Features(rna_X, rna_feature_names, rna_group_dict)
atac_X, atac_feature_names = src.Filter_Features(atac_X, atac_feature_names, atac_group_dict)

D = int(np.sqrt(len(cell_labels)) * np.log(np.log(len(cell_labels))))


train_indices, test_indices = src.Train_Test_Split(cell_labels, seed_obj= np.random.default_rng(replication)) #To keep seed consistent across all experiments with an a dataset

y_train = cell_labels[train_indices]
y_test = cell_labels[test_indices]

print('Estimating Sigma', flush = True)
rna_sigmas = src.Estimate_Sigma(X = rna_X[train_indices,:], group_dict= rna_group_dict, assay ='rna', feature_set= rna_feature_names, 
                                distance_metric= 'euclidean', seed_obj= seed)
atac_sigmas = src.Estimate_Sigma(X = atac_X[train_indices,:], group_dict= atac_group_dict, assay = 'atac', feature_set= atac_feature_names,
                                 distance_metric= 'jaccard', seed_obj= seed)

print('Optimizing Sigma', flush = True)
rna_sigmas = src.Optimize_Sigma(X = rna_X[train_indices,:], y = y_train, group_dict= rna_group_dict, assay = 'rna', D = D, 
                                feature_set= rna_feature_names, sigma_list= rna_sigmas, kernel_type= 'Gaussian', seed_obj= seed,
                                alpha = 1.9, k = 4, sigma_adjustments= np.arange(0.1, 2.1 ,0.3))

atac_sigmas = src.Optimize_Sigma(X = atac_X[train_indices,:], y = y_train, group_dict= atac_group_dict, assay = 'atac', 
                                 D = D, feature_set = atac_feature_names, sigma_list= atac_sigmas, kernel_type= 'Laplacian', seed_obj= seed)


print('Calculating Z', flush = True)
rna_Z_train, rna_Z_test = src.Calculate_Z(X_train = rna_X[train_indices,:], X_test = rna_X[test_indices,:], group_dict= rna_group_dict, assay = 'rna', D = D, 
                                        feature_set= rna_feature_names, sigma_list= rna_sigmas, kernel_type= 'Gaussian', seed_obj= seed)
atac_Z_train, atac_Z_test = src.Calculate_Z(X_train = atac_X[train_indices,:], X_test = atac_X[test_indices,:], group_dict= atac_group_dict, assay = 'atac', D = D, 
                                        feature_set= atac_feature_names, sigma_list= atac_sigmas, kernel_type= 'Laplacian', seed_obj= seed)
del rna_X, atac_X

combined_Z_train, combined_Z_test, group_names = src.Combine_Modalities(Assay_2_name= 'rna', Assay_1_name= 'atac', 
                                                                        Assay_2_Z_train= rna_Z_train, Assay_1_Z_train= atac_Z_train,
                                                                        Assay_2_Group_Names= rna_group_dict.keys(), Assay_1_Group_Names= atac_group_dict.keys(),
                                                                        Assay_2_Z_test= rna_Z_test, Assay_1_Z_test= atac_Z_test)
del rna_Z_train, atac_Z_train
del rna_Z_test, atac_Z_test

_, sparse_alpha = src.Optimize_Alpha(X_train= combined_Z_train, y_train= y_train, group_size= 2*D, starting_alpha=1.9, increment= 0.2, target = 1, n_iter = 10)
_, nonsparse_alpha = src.Optimize_Alpha(X_train= combined_Z_train, y_train= y_train, group_size= 2*D, starting_alpha=1.9, increment= 0.2, target = 100, n_iter = 20)

alpha_list = np.round(np.linspace(sparse_alpha, nonsparse_alpha,10), 2)

metric_dict = {}
selected_pathways = {}
group_norms = {}
predicted = {}
auroc_array = np.zeros(alpha_list.shape)
start = time.time()

for i, alpha in enumerate(alpha_list):
    
    print(f'  Evaluating model. Alpha: {alpha}', flush = True)

    model = src.Train_Model(combined_Z_train, y_train, group_size= 2*D, alpha = alpha)
    predicted[alpha], metric_dict[alpha] = src.Predict(model, combined_Z_test, y_test, metrics = ['AUROC','F1-Score', 'Accuracy', 'Precision', 'Recall'])
    selected_pathways[alpha] = list(src.Find_Selected_Pathways(model, group_names))
    group_norms[alpha] = [np.linalg.norm(model.coef_[i * 2 * D: (i + 1) * 2 * D - 1]) for i in np.arange(len(group_names))]
end = time.time()

results = {}
results['Metrics'] = metric_dict
results['Selected_pathways'] = selected_pathways
results['Norms'] = group_norms
results['Predictions'] = predicted
results['Group_names']= group_names
results['Inference_time'] = end - start
results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'

print(f'Memory Usage: {tracemalloc.get_traced_memory()[1] / 1e9} GB')
tracemalloc.stop()