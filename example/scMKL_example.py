import scmkl
import numpy as np
import scipy

# Setting number of dimensions
D = 100

# Creating AnnData object with cell labels, feature names, grouping dictionary, and data matrix
adata = scmkl.create_adata(X = scipy.sparse.load_npz('./data/MCF7_RNA_X.npz'), 
                         feature_names = np.load('./data/MCF7_RNA_feature_names.npy', allow_pickle = True), 
                         cell_labels = np.load('./data/MCF7_cell_labels.npy', allow_pickle = True), 
                         group_dict = np.load('example/data/RNA_hallmark_groupings.pkl', allow_pickle = True),
                         data_type = 'counts', 
                         split_data = None,
                         D = D, 
                         remove_features = True, 
                         random_state = 100
                         )

# Estimating kernel widths
adata = scmkl.estimate_sigma(adata, n_features = 200)

# Creating Z matrix
adata = scmkl.calculate_z(adata, n_features = 5000)

# Setting a list of alphas to test in cross validation
alpha_list = np.round(np.linspace(2.2,0.05,10), 2)

# Finding best performing alpha value from trainind data
alpha_star = scmkl.optimize_alpha(adata = adata, group_size = 2 * D, tfidf = False, alpha_array = alpha_list, k = 4)

# Training model
adata = scmkl.train_model(adata, group_size= 2 * D, alpha = alpha_star)

# Testing model
predictions, metrics = scmkl.predict(adata, metrics = ['AUROC','F1-Score', 'Accuracy', 'Precision', 'Recall'])

# Finding selected pathways
selected_groups = scmkl.find_selected_groups(adata)

print(metrics)
print(selected_groups)
