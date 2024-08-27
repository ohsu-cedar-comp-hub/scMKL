import numpy as np
import anndata as ad
import gc

from scmkl.tfidf import tfidf_normalize
from scmkl.estimate_sigma import estimate_sigma
from scmkl.approx_kernels import calculate_z


def combine_modalities(assay_1_name: str, assay_2_name: str,
                       assay_1_adata, assay_2_adata,
                       combination = 'concatenate'):
    '''
    Combines data sets for multimodal classification.  Combined group names are assay+group_name
    Input:
            Assay_#_name: Name of assay to be added to group_names as a string if overlap
            Assay_#_adata: Anndata object containing Z matrices and annotations
            combination: How to combine the matrices, either sum or concatenate
    Output:
            combined_adata: Adata object with the combined Z matrices and annotations.  Annotations must match
    '''
    assert assay_1_adata.shape[0] == assay_2_adata.shape[0], 'Cannot combine data with different number of cells.'
    assert assay_1_name != assay_2_name, 'Assay names must be distinct'
    assert combination.lower() in ['sum', 'concatenate']

    assay1_groups = set(list(assay_1_adata.uns['group_dict'].keys()))
    assay2_groups = set(list(assay_2_adata.uns['group_dict'].keys()))

    if not np.all(assay_1_adata.uns['train_indices'] == assay_2_adata.uns['train_indices']):

        assay_1_adata = ad.concat((assay_1_adata[assay_1_adata.uns['train_indices']], assay_1_adata[assay_1_adata.uns['test_indices']]), uns_merge = 'same')
        assay_2_adata = ad.concat((assay_2_adata[assay_2_adata.uns['train_indices']], assay_2_adata[assay_2_adata.uns['test_indices']]), uns_merge = 'same')

        assay_1_adata.uns['train_indices'] = np.arange(len(assay_1_adata.uns['train_indices']))
        assay_1_adata.uns['test_indices'] = np.arange(len(assay_1_adata.uns['train_indices']), 
                                                      len(assay_1_adata.uns['train_indices']) + len(assay_1_adata.uns['test_indices']))
        
        assay_2_adata.uns['train_indices'] = assay_1_adata.uns['train_indices'].copy()
        assay_2_adata.uns['test_indices'] = assay_1_adata.uns['test_indices'].copy()

    combined_adata = ad.concat((assay_1_adata, assay_2_adata), uns_merge = 'same', axis = 1, label = 'labels')
    combined_adata.obs = assay_1_adata.obs

    if 'Z_train' in assay_1_adata.uns and 'Z_train' in assay_2_adata.uns:
        if combination == 'concatenate':
            combined_adata.uns['Z_train'] = np.hstack((assay_1_adata.uns['Z_train'], assay_2_adata.uns['Z_train']))
            combined_adata.uns['Z_test'] = np.hstack((assay_1_adata.uns['Z_test'], assay_2_adata.uns['Z_test']))

        elif combination == 'sum':
            assert assay_1_adata.uns['Z_train'].shape == assay_2_adata.uns['Z_train'].shape, 'Cannot sum Z matrices with different dimensions'
            combined_adata.uns['Z_train'] = assay_1_adata.uns['Z_train'] + assay_2_adata.uns['Z_train']
            combined_adata.uns['Z_test'] = assay_1_adata.uns['Z_test'] + assay_2_adata.uns['Z_test']

    group_dict1 = assay_1_adata.uns['group_dict']
    group_dict2 = assay_2_adata.uns['group_dict']

    if len(assay1_groups.intersection(assay2_groups)) > 0:
        new_dict = {}
        for group, features in group_dict1.items():
            new_dict[f'{assay_1_name}-{group}'] = features
    
        group_dict1 = new_dict

        new_dict = {}
        for group, features in group_dict2.items():
            new_dict[f'{assay_2_name}-{group}'] = features
    
        group_dict2 = new_dict

    group_dict = group_dict1 | group_dict2 #Combines the dictionaries
    combined_adata.uns['group_dict'] = group_dict

    if 'seed_obj' in assay_1_adata.uns_keys():
        combined_adata.uns['seed_obj'] = assay_1_adata.uns['seed_obj']
    else:
        print('No random seed present in Adata, it is recommended for reproducibility.')

    del assay_1_adata, assay_2_adata
    gc.collect()

    return combined_adata


def multimodal_processing(assay1: str, assay2: str ,adata1, adata2, tfidf: list, z_calculation = False):
    '''
    Function to remove rows from both modalities when there is no signal present in at least 1.

    Input:
        assay<N>- Name of assay data contained in adata<N>
        adata<N>- adata object created as above.
        tfidf- list of boolean values whether to tfidf the respective matrices
        z_calculation- Boolean value whether to calculate sigma and Z on the adata before combining
                        Allows for individual kernel functions for each
    Output:
        adata- Concatenated adata objects with empty rows removed and matching order
    '''

    import warnings 
    warnings.filterwarnings('ignore')

    assert adata1.shape[0] == adata2.shape[0], 'Different number of cells present in each object'
    assert np.all(adata1.uns['train_indices'] == adata2.uns['train_indices']), 'Different train indices'
    assert np.all(adata1.uns['test_indices'] == adata2.uns['test_indices']), 'Different test indices'

    non_empty_rows1 = np.where(np.sum(adata1.X, axis = 1) > 0)[0]
    non_empty_rows2 = np.where(np.sum(adata2.X, axis = 1) > 0)[0]

    train_test = np.repeat('train', adata1.shape[0])
    train_test[adata1.uns['test_indices']] = 'test'

    non_empty_rows = np.intersect1d(non_empty_rows1, non_empty_rows2)

    train_test = train_test[non_empty_rows]
    train_indices = np.where(train_test == 'train')[0]
    test_indices = np.where(train_test == 'test')[0]

    adata1.uns['train_indices'] = train_indices
    adata2.uns['train_indices'] = train_indices
    adata1.uns['test_indices'] = test_indices
    adata2.uns['test_indices'] = test_indices

    adata1 = adata1[non_empty_rows, :]
    adata2 = adata2[non_empty_rows, :]

    if tfidf[0]:
        adata1 = tfidf_normalize(adata1, binarize= True)
    if tfidf[1]:
        adata2 = tfidf_normalize(adata2, binarize= True)

    if z_calculation:
        print('Estimating Sigma', flush = True)
        adata1 = estimate_sigma(adata1, n_features= 200)
        adata2 = estimate_sigma(adata2, n_features= 200)
        
        print('Calculating Z', flush = True)
        adata1 = calculate_z(adata1, n_features = 5000)
        adata2 = calculate_z(adata2, n_features= 5000)

    if 'labels' in adata1.obs:
        assert np.all(adata1.obs['labels'] == adata2.obs['labels']), 'Cell labels do not match between adata objects'

    adata = combine_modalities(assay1, assay2, adata1, adata2, 'concatenate')

    del adata1, adata2
    gc.collect()

    return adata    

