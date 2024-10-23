import numpy as np
import anndata as ad
import gc

from scmkl.tfidf import tfidf_normalize
from scmkl.estimate_sigma import estimate_sigma
from scmkl.approx_kernels import calculate_z


def combine_modalities(adatas : list,
                       names : list,
                       combination = 'concatenate'):
    '''
    Combines data sets for multimodal classification.  Combined group names are assay+group_name
    Input:
            adatas: a list of AnnData objects where each object is a different modality
            names: a list of strings names for each modality repective to each object in adatas
            combination: How to combine the matrices, either sum or concatenate
    Output:
            combined_adata: Adata object with the combined Z matrices and annotations.  Annotations must match
    '''
    # assert assay_1_adata.shape[0] == assay_2_adata.shape[0], 'Cannot combine data with different number of cells.'
    assert len(np.unique(names)) == len(names), 'Assay names must be distinct'
    assert combination.lower() in ['sum', 'concatenate']
    assert all([(('Z_train' in adata.uns.keys()) & ('Z_test' in adata.uns.keys())) for adata in adatas]), 'Z not calculated'

    combined_adata = ad.concat(adatas, uns_merge = 'same', axis = 1, label = 'labels')

    assert 'train_indices' in combined_adata.uns.keys(), 'Different train test splits between AnnData objects'

    # Conserving labels from adatas
    combined_adata.obs = adatas[0].obs

    # Creating a single dictionary with all of the groups across modalities 
    group_dict = {}
    for name, adata in zip(names, adatas):
        for group_name, features in adata.uns['group_dict'].items():
            group_dict[f'{name}-{group_name}'] = features

    if combination == 'concatenate':
        combined_adata.uns['Z_train'] = np.hstack([adata.uns['Z_train'] for adata in adatas])
        combined_adata.uns['Z_test'] = np.hstack([adata.uns['Z_test'] for adata in adatas])


    elif combination == 'sum':

        #Check that the dimensions of all Z's are the same before concatentation
        dims = [adata.uns['Z_train'].shape for adata in adatas]
        dims = all([dim == dims[0] for dim in dims])
        assert dims, 'Cannot sum Z matrices with different dimensions'
        
        combined_adata.uns['Z_train'] = np.sum([adata.uns['Z_train'] for adata in adatas], axis = 0)
        combined_adata.uns['Z_test'] = np.sum([adata.uns['Z_test'] for adata in adatas], axis = 0)


    combined_adata.uns['group_dict'] = group_dict

    if 'seed_obj' in adatas[0].uns_keys():
        combined_adata.uns['seed_obj'] = adatas[0].uns['seed_obj']
    else:
        print('No random seed present in Adata, it is recommended for reproducibility.')

    del adatas
    gc.collect()

    return combined_adata


def multimodal_processing(adatas : list, names : list, tfidf: list, z_calculation = False):
    '''
    Function to remove rows from both modalities when there is no signal present in at least 1.

    Input:
        adatas: a list of AnnData objects where each object is a different modality
        names: a list of string names for each modality repective to each object in adatas
        tfidf- list of boolean values whether to tfidf the respective matrices
        z_calculation- Boolean value whether to calculate sigma and Z on the adata before combining
                        Allows for individual kernel functions for each adata
    Output:
        adata- Concatenated adata objects with empty rows removed and matching order
    '''

    import warnings 
    warnings.filterwarnings('ignore')

    assert all([adata.shape[0] for adata in adatas]), 'Different number of cells present in each object'
    assert np.all([np.array_equal(adatas[0].uns['train_indices'], adatas[i].uns['train_indices']) for i in range(1, len(adatas))]), 'Different train indices'
    assert np.all([np.array_equal(adatas[0].uns['test_indices'], adatas[i].uns['test_indices']) for i in range(1, len(adatas))]), 'Different test indices'

    # Getting a list of the rows that are not empty across all of the input modalities
    # Creates a boolean array for each modality of cells with non-empty rows
    non_empty_rows = [np.array((np.sum(adata.X, axis = 1) > 0)).ravel() for adata in adatas]

    # Compares all nonempty boolean arrays and returns a 1d array where sample feature sums
    #   across all modalities are more than 0
    non_empty_rows = np.logical_and(*non_empty_rows).squeeze()

    # Initializing final train test split array
    train_test = np.repeat('train', adatas[0].shape[0])
    train_test[adatas[0].uns['test_indices']] = 'test'

    # Capturing train test split with empty rows filtered out
    train_test = train_test[non_empty_rows]
    train_indices = np.where(train_test == 'train')[0]
    test_indices = np.where(train_test == 'test')[0]

    # Adding train test split arrays to AnnData objects and filtering out empty samples
    for i, adata in enumerate(adatas):
        adata.uns['train_indices'] = train_indices
        adata.uns['test_indices'] = test_indices
        adata = adata[non_empty_rows, :]
        # tfidf normalizing if corresponding element in tfidf is True
        if tfidf[i]:
            adata = tfidf_normalize(adata)

        if z_calculation:
            # AnnData update must be pointing at the object in list
            print(f'Estimating Sigma for {names[i]}', flush = True)
            adatas[i] = estimate_sigma(adata, n_features= 200)
            print(f'Calculating Z for {names[i]}', flush = True)
            adatas[i] = calculate_z(adata, n_features = 5000)

    if 'labels' in adatas[0].obs:
        all_labels = [adata.obs['labels'] for adata in adatas]
        # Ensuring cell labels for each AnnData object are the same
        for i in range(1, len(all_labels)):
            assert np.all(all_labels[0] == all_labels[i]), f'Cell labels between AnnData object in position 0 and position {i} in adatas do not match'

    adata = combine_modalities(adatas = adatas,
                               names = names,
                                combination = 'concatenate')

    del adatas
    gc.collect()

    return adata    

