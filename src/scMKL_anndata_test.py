import numpy as np
import scMKL_src_anndata as src
from scipy.sparse import load_npz
import argparse
import os
import pickle
import time
import tracemalloc

tracemalloc.start()

parser = argparse.ArgumentParser(description='Unimodal classification of single cell data with hallmark prior information')
parser.add_argument('-d', '--dataset', help = 'Which dataset to classify', choices = ['prostate',  'MCF7', 'T47D', 'lymphoma'], type = str)
parser.add_argument('-a', '--assay', help = 'Which assay to use', type = str, choices = ['rna', 'atac', 'gene_scores'])
parser.add_argument('-m', '--distance_metric', help = 'Which of the scipy distance functions to use for estimating sigma', type = str)
parser.add_argument('-r', '--replication', help = 'Which replication to use', type = int, default = 1)
args = parser.parse_args()

dataset = args.dataset
assay = args.assay
replication = args.replication
distance_metric = args.distance_metric

X = load_npz(f'./example/data/{dataset}_{assay.upper()}_X.npz')
feature_names = np.load(f'./example/data/{dataset}_{assay.upper()}_feature_names.npy', allow_pickle= True)
cell_labels = np.load(f'./example/data/{dataset}_cell_labels.npy', allow_pickle= True)

with open(f'./example/data/{dataset}_feature_groupings.pkl', 'rb') as fin:
    groupings = pickle.load(fin)

if assay == 'atac':
    feature_type = 'peaks'
    kernel_func = 'Laplacian'
    dtype = 'binary'
else:
    feature_type = 'genes'
    kernel_func = 'Gaussian'
    dtype = 'counts'

group_dict = {}

for pathway in groupings.keys():
    group_dict[pathway] = groupings[pathway][feature_type]

D = int(np.sqrt(len(cell_labels)) * np.log(np.log(len(cell_labels))))

adata = src.Create_Adata(X = X, feature_names = feature_names, cell_labels = cell_labels, group_dict = group_dict,
                         data_type = dtype, D = D, filter_features = True, random_state = replication)


print('Estimating Sigma', flush = True)
adata = src.Estimate_Sigma(adata, distance_metric = distance_metric, n_features = 200)

print('Optimizing Sigma', flush = True)
adata = src.Optimize_Sigma(adata, kernel_type = kernel_func)

print('Calculating Z', flush = True)
adata = src.Calculate_Z(adata, kernel_type = kernel_func, n_features = 5000)



alpha_list = np.round(np.linspace(1.9,0.1,10), 2)
if dataset == 'prostate':
    alpha_list = np.round(np.linspace(3.6, 0.1, 10), 2)

metric_dict = {}
selected_pathways = {}
group_norms = {}
group_names = list(group_dict.keys())
predicted = {}
auroc_array = np.zeros(alpha_list.shape)
start = time.time()

for i, alpha in enumerate(alpha_list):
    
    print(f'  Evaluating model. Alpha: {alpha}', flush = True)

    adata = src.Train_Model(adata, group_size= 2*D, alpha = alpha)
    predicted[alpha], metric_dict[alpha] = src.Predict(adata, metrics = ['AUROC','F1-Score', 'Accuracy', 'Precision', 'Recall'])
    selected_pathways[alpha] = src.Find_Selected_Pathways(adata)
    group_norms[alpha] = [np.linalg.norm(adata.uns['model'].coef_[i * 2 * D: (i + 1) * 2 * D - 1]) for i in np.arange(len(group_names))]
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


# # output_dir = f'/home/groups/CEDAR/kupp/scMKL/results/scMM/{dataset}/python_MAKL/result_files/unimodal_hallmark'
# # file_name = f'{dataset}_{assay}_hallmark_{distance_metric}_dist_replication_{replication}.pkl'

# # if not os.path.exists(output_dir):
# #     os.makedirs(output_dir)

# # with open(f'{output_dir}/{file_name}', 'wb') as fout:
# #     pickle.dump(results, fout)
