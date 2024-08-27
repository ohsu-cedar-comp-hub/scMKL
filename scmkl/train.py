import numpy as np
import celer
import gc
import tracemalloc


def train_model(adata, group_size = 1, alpha = 0.9):
    '''
    Function to fit a grouplasso model to the provided data.
    Inputs:
            Adata with Z matrices in adata.uns
            group_size- Argument describing how the features are grouped. 
                    From Celer documentation:
                    "groupsint | list of ints | list of lists of ints.
                        Partition of features used in the penalty on w. 
                            If an int is passed, groups are contiguous blocks of features, of size groups. 
                            If a list of ints is passed, groups are assumed to be contiguous, group number g being of size groups[g]. 
                            If a list of lists of ints is passed, groups[g] contains the feature indices of the group number g."
                    If 1, model will behave identically to Lasso Regression
            alpha- Group Lasso regularization coefficient. alpha is a floating point value controlling model solution sparsity. Must be a positive float.
                        The smaller the value, the more feature groups will be selected in the trained model.
    Outputs:

            adata object with trained model in uns accessible with- adata.uns['model']
                Specifics of model:
                    model- The trained Celer Group Lasso model.  Used in Find_Selected_Pathways() function for interpretability.
                        For more information about attributes and methods see the Celer documentation at https://mathurinm.github.io/celer/generated/celer.GroupLasso.html.

    '''
    assert alpha > 0, 'Alpha must be positive'

    y_train = adata.obs['labels'].iloc[adata.uns['train_indices']]
    X_train = adata.uns['Z_train']

    cell_labels = np.unique(y_train)

    # This is a regression algorithm. We need to make the labels 'continuous' for classification, but they will remain binary.
    # Casts training labels to array of -1,1
    train_labels = np.ones(y_train.shape)
    train_labels[y_train == cell_labels[1]] = -1

    # Alphamax is a calculation to regularize the effect of alpha (a sparsity parameter) across different data sets
    alphamax = np.max(np.abs(X_train.T.dot(train_labels))) / X_train.shape[0] * alpha

    # Instantiate celer Group Lasso Regression Model Object
    model = celer.GroupLasso(groups = group_size, alpha = alphamax)

    # Fit model using training data
    model.fit(X_train, train_labels.ravel())

    adata.uns['model'] = model
    return adata