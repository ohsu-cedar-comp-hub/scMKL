import unittest
import scmkl
import numpy as np
import anndata as ad
from test_calculate_z import create_test_adata


def read_h5ad(mod: str):
    """
    Reads in a h5ad file in example/data directory.
    """
    h5_fp = f'../example/data/pbmc_{mod.lower()}.h5ad'
    group_fp = f'../example/data/_{mod.upper()}_azimuth_pbmc_groupings.pkl'
    adata = ad.read_h5ad(h5_fp)

    split_data = np.array(['train']*adata.shape[0])
    split_data[adata.obs['batch'] == '3k'] = 'test'

    adata = scmkl.format_adata(adata, 'celltypes', group_fp, 
                               split_data=split_data, 
                               allow_multiclass=True, 
                               class_threshold=2000)

    return adata


class TestOptimizeAlpha(unittest.TestCase):
    """
    This unittest class is used to ensure that scmkl.optimize_alpha() 
    works properly for both unimodal and multimodal runs.
    """
    def test_unimodal_optimize_alpha(self):
        """
        This function ensure that scmkl.optimize_alpha works correctly 
        for unimodal test cases by checking that the output is what we 
        expect.
        """
        adata = create_test_adata()
        alpha_list = np.array([0.01, 0.05, 0.1])

        # Finding optimal alpha
        alpha_star = scmkl.optimize_alpha(adata, 
                                          alpha_array=alpha_list, 
                                          batch_size=60)

        # Checking that optimal alpha is what we expect
        self.assertEqual(alpha_star, 0.1, 
                         ("scmkl.optimize_alpha chose the wrong alpha "
                          "as optimal for unimodal"))
        

    def test_multimodal_optimize_alpha(self):
        """
        This function ensure that scmkl.optimize_alpha works correctly 
        for multimodal test cases by checking that the output is what 
        we expect.
        """
        adatas = [create_test_adata('RNA'), create_test_adata('ATAC')]
        alpha_list = np.array([0.01, 0.05, 0.1])

        alpha_star = scmkl.optimize_alpha(adatas, alpha_array=alpha_list,
                                          batch_size=60)

        self.assertEqual(alpha_star, 0.1, 
                         ("scmkl.optimize_alpha chose the wrong alpha "
                          "as optimal for multimodal run"))

    
    def test_multiclass_optimize_alpha(self):
        """
        This function ensures finding the optimal alpha for best 
        performance via cross validation for multiclass runs is working 
        properly.
        """
        adata = read_h5ad('RNA')
        alphas = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        train_dict = scmkl.get_class_train(adata.uns['train_indices'],
                                           adata.obs['labels'], 
                                           adata.uns['seed_obj'])

        alpha_dict = scmkl.optimize_alpha(adata, 
                                          alpha_array=alphas, 
                                          batch_size=26, 
                                          train_dict=train_dict,
                                          metric='F1-Score')

        expected_alphas = {
            'B': 0.1, 
            'CD14+ Monocytes': 0.1, 
            'CD16+ Monocytes': 0.1, 
            'CD4 T': 0.1, 
            'CD8 T': 0.1, 
            'Dendritic': 0.1, 
            'NK': 0.1}

        self.assertDictEqual(alpha_dict, expected_alphas, 
                             ("scmkl.optimize_alpha() for multiclass "
                              "runs returned incorrect values."))
        

    def test_multiview_multiclass_optimize_alpha(self):
        """
        This function ensures that in a multiview and multclass 
        optimize alpha task, everything runs correctly.
        """
        adatas = [read_h5ad('RNA'), read_h5ad('ATAC')]
        alphas = np.array([0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8])

        train_dict = scmkl.get_class_train(adatas[0].uns['train_indices'],
                                           adatas[0].obs['labels'], 
                                           adatas[0].uns['seed_obj'])

        alpha_dict = scmkl.optimize_alpha(adatas, 
                                           alpha_array=alphas, 
                                           train_dict=train_dict,
                                           metric='F1-Score',
                                           batch_size=26)
        
        expected_alphas = {
            'B': 0.3, 
            'CD14+ Monocytes': 0.3, 
            'CD16+ Monocytes': 0.1, 
            'CD4 T': 0.05, 
            'CD8 T': 0.3, 
            'Dendritic': 0.05, 
            'NK': 0.1}
        
        self.assertDictEqual(alpha_dict, expected_alphas, 
                             ("scmkl.optimize_alpha() for multiclass "
                              "runs returned incorrect values."))
        
    def test_folds_and_processing(self):
        """
        Much of this code adapted from optimize alpha code
        """
        # Need a matrix to track fold generation with
        x = np.arange(500).reshape(500,1)
        x = np.concatenate([x, x, x], axis=1)
        vars = {f'gene_{i}' for i in range(x.shape[1])}
        labs = ['class_1', 'class_2', 'class_2', 'class_1', 'class_1']*100

        group_dict = {
            'Group1': ['gene_0', 'gene_2'],
            'Group2': ['gene_0', 'gene_1']
        }

        train_test = np.array(300*['train'] + 200*['test'])

        adata = scmkl.create_adata(x, vars, labs, 
                                    group_dict, 
                                    scale_data=False,
                                    split_data=train_test)

        adata = [adata]

        k = 4

        # Only want folds for training samples
        train_indices = adata[0].uns['train_indices'].copy()
        cv_adata = [adata[i][train_indices, :].copy()
                    for i in range(len(adata))]

        folds = scmkl.get_folds(adata[0], k)

        for fold in range(k):

            fold_train, fold_test = folds[fold]
            fold_adata = cv_adata.copy()

            # Downstream functions expect train then test samples in adata(s)
            sort_idcs, fold_train, fold_test = scmkl.sort_samples(fold_train, fold_test)
            fold_adata = scmkl.prepare_fold(fold_adata, sort_idcs, 
                                            fold_train, fold_test)
                
            names = ['Adata ' + str(i + 1) for i in range(len(cv_adata))]

            # Adatas need combined if applicable and kernels calculated 
            fold_adata = scmkl.process_fold(fold_adata, names, tfidf=[False], combination='concatenate', 
                                            batches=10, batch_size=10)

            # Each sample contains only values cooresponding to orig idx
            exp_order = np.concatenate(folds[fold])
            exp_mat = np.repeat(exp_order.reshape(cv_adata[0].shape[0], 1), 
                                cv_adata[0].shape[1], 
                                axis=1)

            # Repeating indices should equal mat (see x initialization)
            sort_err = ("After processing, matrix no longer "
                        "cooresponds with expected indices")
            self.assertTrue(np.array_equal(fold_adata.X, exp_mat), sort_err)

if __name__ == '__main__':
    unittest.main()
