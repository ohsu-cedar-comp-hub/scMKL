import anndata as ad
import numpy as np


def _check_adatas(adatas : list, check_uns : bool = False, 
                  check_obs : bool = False) -> None:
    """
    Takes a list of AnnData objects and checks that all training and
    testing indices stored in adata.uns and cell labels stored in 
    adata.obs are the same. Ensures that all objects are of type 
    AnnData.
    Args:
        adatas - a list of AnnData objects where each element has the
            same array of indices for adata.uns['train_indices'] and 
            the same array of indices for adata.uns-'test_indices'].
            Additionally, adata.obs['labels'] must be the same for all
            AnnData objects.
        check_uns - a boolean value. If True, function will ensure that
            'train_indices' and 'test_indices' in adata.uns are the
            same. If False, indices will not be checked.
        check_obs - a boolean value. If True, function will ensure that
            adata.obs['labels'] in each AnnData object are the same.
    Returns:
        Returns None. Will throw an error if not all of the criteria
        listed above are met by adatas.
    """
    for i, adata in enumerate(adatas):
        # Ensuring all elements are type AnnData
        if type(adata) != ad.AnnData: 
            raise TypeError("All elements in adatas should be of type "
                            "anndata.AnnData")
        
        # Ensuring all train/test indices are the same
        if check_uns:
            assert np.array_equal(adatas[0].uns['train_indices'],
                                  adata.uns['train_indices']), ("Train "
                                    "indices across AnnData objects in "
                                    "adatas do not match, ensure "
                                    "adata.uns['train_indices'] are the same "
                                    "for all elements")
                
            assert np.array_equal(adatas[0].uns['test_indices'], 
                                    adata.uns['test_indices']), ("Test "
                                    "indices across AnnData objects in "
                                    "adatas do not match, ensure "
                                    "adata.uns['test_indices'] are the same "
                                    "for all elements")

        # Ensuring all cell labels are the same 
        if check_obs:
            assert np.array_equal(adatas[0].obs['labels'], 
                                    adata.obs['labels']), (
                                    "adata.obs['labels'] are different "
                                    "between AnnData objects")
    
    return None