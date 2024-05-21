import numpy as np
from scipy.sparse import load_npz
import argparse
import os
import pickle
import time
import tracemalloc
import sys
sys.path.append('./src/')
import scMKL_src as src

tracemalloc.start()


parser = argparse.ArgumentParser(description='Unimodal classification of single cell data with hallmark prior information')
# parser.add_argument('-d', '--dataset', help = 'Which dataset to classify', choices = ['prostate',  'MCF7', 'T47D', 'lymphoma'], type = str)
parser.add_argument('-a', '--assay', help = 'Which assay to use', type = str, choices = ['rna', 'atac', 'gene_scores'])
parser.add_argument('-r', '--replication', help = 'Which replication to use', type = int, default = 1)
parser.add_argument('-m', '--distance_metric', help = 'Which Scipy distance function to use to estimate sigma', type = str, default = 'euclidean')
args = parser.parse_args()

# dataset = args.dataset
assay = args.assay
replication = args.replication
distance_metric = args.distance_metric

replication = 1
seed_obj = np.random.default_rng(100*replication)

X = load_npz(f'example/data/MCF7_{assay.upper()}_X.npz')
feature_names = np.load(f'example/data/MCF7_{assay.upper()}_feature_names.npy', allow_pickle= True)
cell_labels = np.load(f'example/data/MCF7_cell_labels.npy', allow_pickle= True)

with open(f'example/data/MCF7_feature_groupings.pkl', 'rb') as fin:
    feature_groupings = pickle.load(fin)

if assay == 'atac':
    feature_type = 'peaks'
    kernel_func = 'Laplacian'
else:
    feature_type = 'genes'
    kernel_func = 'Gaussian'

group_dict = {}
for pathway in feature_groupings.keys():
    group_dict[pathway] = feature_groupings[pathway][feature_type]

D = int(np.sqrt(len(cell_labels)) * np.log(np.log(len(cell_labels))))

X, feature_names = src.Filter_Features(X, feature_names, group_dict)

train_indices, test_indices = src.Train_Test_Split(cell_labels, seed_obj= np.random.default_rng(replication)) #To keep seed consistent across all experiments with an a dataset

X_train = X[train_indices,:]
y_train = cell_labels[train_indices]
X_test = X[test_indices,:]
y_test = cell_labels[test_indices]

print('Estimating Sigma', flush = True)
sigmas = src.Estimate_Sigma(X = X_train, group_dict= group_dict, assay= assay, feature_set= feature_names, distance_metric= distance_metric, seed_obj= seed_obj)

print('Optimizing Sigma', flush = True)
# sigmas = src.Optimize_Sigma(X_train, y_train, group_dict, assay, D, feature_names, sigmas, kernel_func, seed_obj)

print('Calculating Z', flush = True)
Z_train, Z_test = src.Calculate_Z(X_train, X_test, group_dict, assay, D, feature_names, sigmas, kernel_func, seed_obj)

print('Finding Alpha', flush = True)
_, sparse_alpha = src.Optimize_Alpha(X_train= Z_train, y_train= y_train, group_size= 2*D, starting_alpha=1.9, increment= 0.2, target = 1, n_iter = 10)
_, nonsparse_alpha = src.Optimize_Alpha(X_train= Z_train, y_train= y_train, group_size= 2*D, starting_alpha=1.9, increment= 0.2, target = 50, n_iter = 10)

alpha_list = np.round(np.linspace(sparse_alpha, nonsparse_alpha,10), 2)

metric_dict = {}
selected_pathways = {}
group_norms = {}
group_names = list(group_dict.keys())
predicted = {}
auroc_array = np.zeros(alpha_list.shape)
start = time.time()

for i, alpha in enumerate(alpha_list):
    
    print(f'  Evaluating model. Alpha: {alpha}', flush = True)

    model = src.Train_Model(Z_train, y_train, group_size= 2*D, alpha = alpha)
    auroc_array[i] = src.Calculate_AUROC(model, Z_test, y_test)
    predicted[alpha], metric_dict[alpha] = src.Predict(model, Z_test, y_test, metrics = ['AUROC','F1-Score', 'Accuracy', 'Precision', 'Recall'])
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