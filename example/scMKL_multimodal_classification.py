import numpy as np
import scMKL_src as src
from scipy.sparse import load_npz
import argparse
import os
import pickle
import time
import tracemalloc

tracemalloc.start()

parser = argparse.ArgumentParser(description='Unimodal classification of single cell data with hallmark prior information')
parser.add_argument('-d', '--dataset', help = 'Which dataset to classify', choices = ['prostate',  'MCF7', 'T47D', 'lymphoma'], type = str)
parser.add_argument('-r', '--replication', help = 'Which replication to use', type = int, default = 1)
args = parser.parse_args()

dataset = args.dataset
replication = args.replication

seed_obj = np.random.default_rng(100*replication)

rna_X = load_npz(f'/home/groups/CEDAR/kupp/scMKL/data/scMM/{dataset}/{dataset}_RNA_scipy.npz')
rna_feature_names = np.load(f'/home/groups/CEDAR/kupp/scMKL/data/scMM/{dataset}/{dataset}_RNA_feature_names.npy', allow_pickle= True)

atac_X = load_npz(f'/home/groups/CEDAR/kupp/scMKL/data/scMM/{dataset}/{dataset}_ATAC_scipy.npz')
atac_feature_names = np.load(f'/home/groups/CEDAR/kupp/scMKL/data/scMM/{dataset}/{dataset}_ATAC_feature_names.npy', allow_pickle= True)

cell_labels = np.load(f'/home/groups/CEDAR/kupp/scMKL/data/scMM/{dataset}/{dataset}_cell_metadata.npy', allow_pickle= True)

sample = seed_obj.choice(np.arange(len(cell_labels)), 1000)
rna_X = rna_X[sample,:]
atac_X = atac_X[sample,:]
cell_labels = cell_labels[sample]

with open(f'/home/groups/CEDAR/kupp/scMKL/data/scMM/{dataset}/{dataset}_groupings.pkl', 'rb') as fin:
    groupings = pickle.load(fin)

rna_group_dict = {}
atac_group_dict = {}

for pathway in groupings['hallmark'].keys():
    rna_group_dict[pathway] = groupings['hallmark'][pathway]['genes']
    atac_group_dict[pathway] = groupings['hallmark'][pathway]['peaks']

rna_X, rna_feature_names = src.Filter_Features(rna_X, rna_feature_names, rna_group_dict)
atac_X, atac_feature_names = src.Filter_Features(atac_X, atac_feature_names, atac_group_dict)

D = int(np.sqrt(len(cell_labels)) * np.log(np.log(len(cell_labels))))


train_indices, test_indices = src.Train_Test_Split(cell_labels, seed_obj= np.random.default_rng(replication)) #To keep seed consistent across all experiments with an a dataset


print('Estimating Sigma', flush = True)
rna_sigmas = src.Estimate_Sigma(X = rna_X[train_indices,:], group_dict= rna_group_dict, assay ='rna', feature_set= rna_feature_names, 
                                distance_metric= 'euclidean', seed_obj= seed_obj)
atac_sigmas = src.Estimate_Sigma(X = atac_X[train_indices,:], group_dict= atac_group_dict, assay = 'atac', feature_set= atac_feature_names,
                                 distance_metric= 'jaccard', seed_obj= seed_obj)

print('Optimizing Sigma', flush = True)
rna_sigmas = src.Optimize_Sigma(X = rna_X[train_indices,:], y = cell_labels[train_indices], group_dict= rna_group_dict, assay = 'rna', D = D, 
                                feature_set= rna_feature_names, sigma_list= rna_sigmas, kernel_type= 'Gaussian', seed_obj= seed_obj,
                                alpha = 1.9, k = 4, sigma_adjustments= np.arange(0.1, 2.1 ,0.3))

atac_sigmas = src.Optimize_Sigma(X = atac_X[train_indices,:], y = cell_labels[train_indices], group_dict= atac_group_dict, assay = 'atac', 
                                 D = D, feature_set = atac_feature_names, sigma_list= atac_sigmas, kernel_type= 'Laplacian', seed_obj= seed_obj)


print('Calculating Z', flush = True)
rna_Z_train, rna_Z_test = src.Calculate_Z(X_train = rna_X[train_indices,:], X_test = rna_X[test_indices,:], group_dict= rna_group_dict, assay = 'rna', D = D, 
                                        feature_set= rna_feature_names, sigma_list= rna_sigmas, kernel_type= 'Gaussian', seed_obj= seed_obj)
atac_Z_train, atac_Z_test = src.Calculate_Z(X_train = atac_X[train_indices,:], X_test = atac_X[test_indices,:], group_dict= atac_group_dict, assay = 'atac', D = D, 
                                        feature_set= atac_feature_names, sigma_list= atac_sigmas, kernel_type= 'Laplacian', seed_obj= seed_obj)
del rna_X, atac_X

combined_Z_train, combined_Z_test, group_names = src.Combine_Modalities(Assay_1_name= 'rna', Assay_2_name= 'atac', 
                                                                        Assay_1_Z_train= rna_Z_train, Assay_2_Z_train= atac_Z_train,
                                                                        Assay_1_Group_Names= rna_group_dict.keys(), Assay_2_Group_Names= atac_group_dict.keys(),
                                                                        Assay_1_Z_test= rna_Z_test, Assay_2_Z_test= atac_Z_test)
del rna_Z_train, atac_Z_train
del rna_Z_test, atac_Z_test

alpha_list = np.round(np.linspace(1.9,0.1,10), 2)
if dataset == 'prostate':
    alpha_list = np.round(np.linspace(3.6, 0.9, 10), 2)

metric_dict = {}
selected_pathways = {}
group_norms = {}
predicted = {}
auroc_array = np.zeros(alpha_list.shape)
start = time.time()

for i, alpha in enumerate(alpha_list):
    
    print(f'  Evaluating model. Alpha: {alpha}', flush = True)

    model = src.Train_Model(combined_Z_train, cell_labels[train_indices], group_size= 2*D, alpha = alpha)
    predicted[alpha], metric_dict[alpha] = src.Predict(model, combined_Z_test, cell_labels[test_indices], metrics = ['AUROC','F1-Score', 'Accuracy', 'Precision', 'Recall'])
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