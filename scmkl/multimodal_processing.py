import numpy as np
import anndata as ad
import gc

from scmkl.tfidf_normalize import tfidf_normalize
from scmkl.data_processing import sparse_var
from scmkl.calculate_z import calculate_z


def combine_modalities(adatas: list[ad.AnnData], names: list[str], 
                        combination: str = 'concatenate'):
    """
    Combines data sets for multimodal classification. Combined group 
    names are `f'{assay}+{group_name}'`.

    Parameters
    ----------
    adatas : list[ad.AnnData]
        List of AnnData objects where each object is a different 
        modality. Annotations must match between objects (i.e. same 
        sample order).

    names : list[str]
        List of strings names for each modality repective to each 
        object in adatas.
            
    combination: str
        How to combine the matrices, either `'sum'` or `'concatenate'`.
    
    Returns
    -------
    combined_adata : ad.Anndata
        Adata object with the combined Z matrices and annotations. 
    """
    assert len({adata.shape[0] for adata in adatas}) == 1, ("All adatas must "
                                                            "have the same "
                                                            "number of rows")
    assert len(np.unique(names)) == len(names), "Assay names must be distinct"
    assert combination.lower() in ['sum', 'concatenate']

    z_train = all(['Z_train' in adata.uns.keys() for adata in adatas])
    z_test = all(['Z_test' in adata.uns.keys() for adata in adatas])

    assert all([z_train, z_test]), "Z not calculated for one or more adatas"

    # Combining modalities
    combined_adata = ad.concat(adatas, uns_merge = 'same', 
                               axis = 1, label = 'labels')
    
    assert 'train_indices' in combined_adata.uns.keys(), ("Different train "
                                                          "test splits "
                                                          "between AnnData "
                                                          "objects")

    # Conserving labels from adatas
    combined_adata.obs = adatas[0].obs

    # Creating a single dictionary with all of the groups across modalities 
    group_dict = {}
    for name, adata in zip(names, adatas):
        for group_name, features in adata.uns['group_dict'].items():
            group_dict[f'{name}-{group_name}'] = features

    if combination == 'concatenate':
        combined_adata.uns['Z_train'] = np.hstack([adata.uns['Z_train'] 
                                                   for adata in adatas])
        combined_adata.uns['Z_test'] = np.hstack([adata.uns['Z_test'] 
                                                  for adata in adatas])


    elif combination == 'sum':

        #Check that the dimensions of all Z's are the same
        dims = [adata.uns['Z_train'].shape for adata in adatas]
        dims = all([dim == dims[0] for dim in dims])
        assert dims, "Cannot sum Z matrices with different dimensions"
        
        combined_adata.uns['Z_train'] = np.sum([adata.uns['Z_train'] 
                                                for adata in adatas], 
                                                axis = 0)
        combined_adata.uns['Z_test'] = np.sum([adata.uns['Z_test'] 
                                               for adata in adatas], 
                                               axis = 0)


    combined_adata.uns['group_dict'] = group_dict

    if 'seed_obj' in adatas[0].uns_keys():
        combined_adata.uns['seed_obj'] = adatas[0].uns['seed_obj']
    else:
        print("No random seed present in adata"
              "Recommended for reproducibility.")

    del adatas
    gc.collect()

    return combined_adata


def multimodal_processing(adatas : list[ad.AnnData], names : list[str], 
                          tfidf: list[bool], combination: str='concatenate', 
                          batches: int=10, batch_size: int=100) -> ad.AnnData:
    """
    Combines and processes a list of `ad.AnnData` objects.

    Parameters
    ----------
    adatas : list[ad.AnnData]
        List of `ad.AnnData` objects where each object is a different 
        modality. Annotations must match between objects (i.e. same 
        sample order).

    names : list[str]
        List of strings names for each modality repective to each 
        object in adatas.
            
    combination: str
        How to combine the matrices, either `'sum'` or `'concatenate'`.
    
    tfidf : list[bool]
        If element `i` is `True`, `adata[i]` will be TF-IDF normalized.

    batches : int
        The number of batches to use for the distance calculation.
        This will average the result of `batches` distance calculations
        of `batch_size` randomly sampled cells. More batches will converge
        to population distance values at the cost of scalability.

    batch_size : int
        The number of cells to include per batch for distance
        calculations. Higher batch size will converge to population
        distance values at the cost of scalability.
        If `batches*batch_size > num_training_cells`, `batch_size` 
        will be reduced to `int(num_training_cells / batches)`.

    Returns
    -------
    adata : ad.AnnData
        Concatenated from objects from `adatas` with Z matrices 
        calculated.

    Examples
    --------
    >>> rna_adata = scmkl.create_adata(X = mcf7_rna_mat, 
    ...                                feature_names = gene_names, 
    ...                                scale_data = True, 
    ...                                cell_labels = cell_labels, 
    ...                                 group_dict = rna_grouping)
    >>>
    >>> atac_adata = scmkl.create_adata(X = mcf7_atac_mat, 
    ...                                 feature_names = peak_names, 
    ...                                 scale_data = False, 
    ...                                 cell_labels = cell_labels, 
    ...                                 group_dict = atac_grouping)
    >>>
    >>> adatas = [rna_adata, atac_adata]
    >>> mod_names = ['rna', 'atac']
    >>> adata = scmkl.multimodal_processing(adatas = adatas, 
    ...                                     names = mod_names,
    ...                                     tfidf = [False, True])
    >>>
    >>> adata
    AnnData object with n_obs × n_vars = 1000 × 12676
    obs: 'labels'
    var: 'labels'
    uns: 'D', 'kernel_type', 'distance_metric', 'train_indices',  
    'test_indices', 'Z_train', 'Z_test', 'group_dict', 'seed_obj'
    """
    import warnings 
    warnings.filterwarnings('ignore')

    diff_num_warn = "Different number of cells present in each object."
    assert all([adata.shape[0] for adata in adatas]), diff_num_warn
    
    # True if all train indices match
    same_train = np.all([np.array_equal(adatas[0].uns['train_indices'], 
                                        adatas[i].uns['train_indices']) 
                         for i in range(1, len(adatas))])

    # True if all test indices match
    same_test = np.all([np.array_equal(adatas[0].uns['test_indices'], 
                                       adatas[i].uns['test_indices']) 
                        for i in range(1, len(adatas))])

    assert same_train, "Different train indices"
    assert same_test, "Different test indices"

    # Creates a boolean array for each modality of cells with non-empty rows
    non_empty_rows = [np.array(sparse_var(adata.X, axis = 1) != 0).ravel() 
                      for adata in adatas]

    # Returns a 1d array where sample feature sums
    # across all modalities are more than 0
    non_empty_rows = np.logical_and(*non_empty_rows).squeeze()

    # Initializing final train test split array
    train_test = np.repeat('train', adatas[0].shape[0])
    train_test[adatas[0].uns['test_indices']] = 'test'

    # Capturing train test split with empty rows filtered out
    train_test = train_test[non_empty_rows]
    train_indices = np.where(train_test == 'train')[0]
    test_indices = np.where(train_test == 'test')[0]

    # Adding train test split arrays to AnnData objects 
    # and filtering out empty samples
    for i, adata in enumerate(adatas):
        adatas[i].uns['train_indices'] = train_indices
        adatas[i].uns['test_indices'] = test_indices
        adatas[i] = adata[non_empty_rows, :]
        # tfidf normalizing if corresponding element in tfidf is True
        if tfidf[i]:
            adatas[i] = tfidf_normalize(adata)

        print(f"Estimating sigma and calculating Z for {names[i]}", flush = True)
        adatas[i] = calculate_z(adata, n_features = 5000, batches=batches, 
                                batch_size=batch_size)

    if 'labels' in adatas[0].obs:
        all_labels = [adata.obs['labels'] for adata in adatas]
        # Ensuring cell labels for each AnnData object are the same
        uneq_labs_warn = ("Cell labels between AnnData object in position 0 "
                          "and position {} in adatas do not match")
        for i in range(1, len(all_labels)):
            same_labels = np.all(all_labels[0] == all_labels[i])
            assert same_labels, uneq_labs_warn.format(i)

    adata = combine_modalities(adatas=adatas,
                                names=names,
                                combination=combination)

    del adatas
    gc.collect()

    return adata    
