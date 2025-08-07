import numpy as np
import celer
import anndata as ad


def train_model(adata: ad.AnnData, group_size: int | None=None, alpha:float=0.9):
    """
    Fit a grouplasso model to the provided data.

    Parameters
    ----------
    adata : ad.AnnData 
        Has `'Z_train'` and `'Z_test'` keys in `.uns.keys()`.

    group_size : None | int
        Argument describing how the features are grouped. If `None`, 
        `2 * adata.uns['D']` will be used. For more information see 
        [celer documentation](https://mathurinm.github.io/celer/
        generated/celer.GroupLasso.html).
            
    alpha : float
        Group Lasso regularization coefficient, is a floating point 
        value controlling model solution sparsity. Must be a positive 
        float. The smaller the value, the more feature groups will be 
        selected in the trained model.
    
    Returns
    -------
    adata : ad.AnnData 
        Trained model accessible with `adata.uns['model']`.

    Examples
    --------
    >>> adata = scmkl.estimate_sigma(adata)
    >>> adata = scmkl.calculate_z(adata)
    >>> metrics = ['AUROC', 'F1-Score', 'Accuracy', 'Precision', 
    ...            'Recall']
    >>> d = scmkl.calculate_d(adata.shape[0])
    >>> group_size = 2 * d
    >>> adata = scmkl.train_model(adata, group_size)
    >>>
    >>> 'model' in adata.uns.keys()
    True

    See Also
    --------
    celer :
        https://mathurinm.github.io/celer/generated/celer.GroupLasso.html
    """
    assert alpha > 0, 'Alpha must be positive'

    if group_size is None:
        group_size = 2*adata.uns['D']

    y_train = adata.obs['labels'].iloc[adata.uns['train_indices']]
    X_train = adata.uns['Z_train'][adata.uns['train_indices']]

    cell_labels = np.unique(y_train)

    # This is a regression algorithm. We need to make the labels 'continuous' 
    # for classification, but they will remain binary. Casts training labels 
    # to array of -1,1
    train_labels = np.ones(y_train.shape)
    train_labels[y_train == cell_labels[1]] = -1

    # Alphamax is a calculation to regularize the effect of alpha across 
    # different data sets
    alphamax = np.max(np.abs(X_train.T.dot(train_labels)))
    alphamax /= X_train.shape[0] 
    alphamax *= alpha

    # Instantiate celer Group Lasso Regression Model Object
    model = celer.GroupLasso(groups = group_size, alpha = alphamax)

    # Fit model using training data
    model.fit(X_train, train_labels.ravel())

    adata.uns['model'] = model
    return adata