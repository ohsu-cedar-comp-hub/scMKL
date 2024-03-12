# scMKL

This is an introduction to single cell Multiple Kernel Learning. scMKL is a classification algorithm utilizing prior information to group features to enhance classification and aid understanding of distinguishing features in multi-omic data sets.

```
# Packages needed to import data
import numpy as np
import pickle
import sys



# This sys command allows us to import the scMKL_src module from any directory. '..' can be replaced by any path to the module
sys.path.insert(0, '..')
import src.scMKL_src as src
```

#### Inputs for scMKL

There are 4 required pieces of data (per modality) required for scMKL

- The data matrix itself with cells as rows and features as columns.
    - Can be either a Numpy Array or Scipy Sparse array (scipy.sparse.csc_array is the recommended format).
- The sample labels in a Numpy Array. To perform group lasso, these labels must be binary.
- Feature names in a Numpy Array. These are the names of the features corresponding with the data matrix
- A dictionary with grouping data. The keys are the names of the groups, and the values are the corresponding features.
    - Example: {Group1: [feature1, feature2, feature3], Group2: [feature4, feature5, feature6], ...}

```
x = np.load('./data/TCGA-ESCA.npy', allow_pickle = True)
labels = np.load('./data/TCGA-ESCA_cell_metadata.npy', allow_pickle = True)
features = np.load('./data/TCGA-ESCA_RNA_feature_names.npy', allow_pickle = True)


with open('./data/RNA_hallmark_groupings.pkl', 'rb') as fin:
    group_dict = pickle.load(fin)

# This value for D, the number of fourier features in Z, was found to be optimal in previous literature.  Generally increasing D increases accuracy, but runs slower.
D = int(np.sqrt(len(labels)) * np.log(np.log(len(labels))))
```

#### Parameter Optimization

Kernel widths (sigma) are a parameter of the kernel approximation. Here we calculate sigma on the full dataset before optimizing it with k-Fold Cross Validation on the training set.

```
sigmas = src.Calculate_Sigma(x, group_dict, 'rna', features)

# The train/test sets are calculated to keep the proportion of each label the same in the training and testing sets.
train_indices, test_indices = src.Train_Test_Split(labels, seed_obj= np.random.RandomState(500))

x_train = x[train_indices,:]
x_test = x[test_indices,:]
y_train = labels[train_indices]
y_test = labels[test_indices]


sigmas = src.Optimize_Sigma(X = x_train, y = y_train, group_dict = group_dict, assay = 'rna', D = D, feature_set = features, sigma_list = sigmas)
```

#### Calculating Z and Model Evaluation

Below, we calculate approximate kernels for each group in the grouping information.

Then we evaluate the model to find the Area Under the Receiver Operating Curve, and view the distinguishing features between groups.

```
Z_train, Z_test = src.Calculate_Z(x_train, x_test, group_dict, 'rna', D, features, sigmas)

gl = src.Train_Model(Z_train, y_train, 2 * D)
auc = src.Calculate_Auroc(gl, Z_test, y_test)
selected_groups = src.Find_Selected_Pathways(gl, group_names= group_dict.keys())
```


