### scMKL
This is an introduction to single cell Multiple Kernel Learning. scMKL is a classification algorithm utilizing prior information to group features to enhance classification and aid understanding of distinguishing features in multi-omic data sets.


```python
# Packages needed to import data
import numpy as np
import pickle
import sys

# This sys command allows us to import the scMKL_src module from any directory. '..' can be replaced by any path to the module
sys.path.insert(0, '..')
import src.scMKL_src as src

seed = np.random.default_rng(1)
```

#### Inputs for scMKL
There are 4 required pieces of data (per modality) required for scMKL
- The data matrix itself with cells as rows and features as columns.
    - Can be either a Numpy Array or Scipy Sparse array.  
- The sample labels in a Numpy Array.  To perform group lasso, these labels must be binary.
- Feature names in a Numpy Array. These are the names of the features corresponding with the data matrix
- A dictionary with grouping data.  The keys are the names of the groups, and the values are the corresponding features.
    - Example: {Group1: [feature1, feature2, feature3], Group2: [feature4, feature5, feature6], ...}


```python
X = np.load('./data/TCGA-ESCA.npy', allow_pickle = True)
labels = np.load('./data/TCGA-ESCA_cell_metadata.npy', allow_pickle = True)
features = np.load('./data/TCGA-ESCA_RNA_feature_names.npy', allow_pickle = True)


with open('./data/RNA_hallmark_groupings.pkl', 'rb') as fin:
    group_dict = pickle.load(fin)

# This value for D, the number of fourier features in Z, was found to be optimal in previous literature.  Generally increasing D increases accuracy, but runs slower.
D = int(np.sqrt(len(labels)) * np.log(np.log(len(labels))))

# Removes features in X and features that are not found in group_dict.  Done to reduce memory usage and search time
X, features = src.Filter_Features(X, features, group_dict)
```

#### Parameter Optimization
Kernel widths (sigma) are a parameter of the kernel approximation.  Here we estimate sigma on a random 2000 samples from the training set before optimizing it with k-Fold Cross Validation on the full training set.


```python
# The train/test sets are calculated to keep the proportion of each label the same in the training and testing sets.
train_indices, test_indices = src.Train_Test_Split(labels, seed_obj= seed)

X_train = X[train_indices,:]
X_test = X[test_indices,:]
y_train = labels[train_indices]
y_test = labels[test_indices]

sigmas = src.Estimate_Sigma(X= X_train, group_dict= group_dict, assay= 'rna', feature_set= features, seed_obj= seed)

sigmas = src.Optimize_Sigma(X = X_train, y = y_train, group_dict = group_dict, assay = 'rna', D = D, feature_set = features, 
                            sigma_list = sigmas, k = 2, sigma_adjustments = np.arange(0.1,2,0.1), seed_obj= seed)
```

#### Calculating Z and Model Evaluation
Below, we calculate approximate kernels for each group in the grouping information.

Then we train the model to view the distinguishing feature groups between phenotypes and evaluate on a test set to quantify classification performance.
Looking at group normalized weights can reveal insights into underlying biology.


```python
Z_train, Z_test = src.Calculate_Z(X_train= X_train, X_test= X_test, group_dict= group_dict, assay= 'rna', D= D, feature_set= features, sigma_list= sigmas, seed_obj= seed)

gl = src.Train_Model(Z_train, y_train, 2 * D)
predictions, metrics = src.Predict(gl, Z_test, y_test, metrics = ['AUROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall'])
selected_groups = src.Find_Selected_Pathways(gl, group_names= group_dict.keys())
```


```python
print(metrics)
print(selected_groups)
```

    {'AUROC': 1.0, 'Accuracy': 1.0, 'F1-Score': 1.0, 'Precision': 1.0, 'Recall': 1.0}
    ['HALLMARK_ESTROGEN_RESPONSE_EARLY']

