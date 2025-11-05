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
    # def test_unimodal_optimize_alpha(self):
    #     """
    #     This function ensure that scmkl.optimize_alpha works correctly 
    #     for unimodal test cases by checking that the output is what we 
    #     expect.
    #     """
    #     adata = create_test_adata()
    #     alpha_list = np.array([0.01, 0.05, 0.1])

    #     # Finding optimal alpha
    #     alpha_star = scmkl.optimize_alpha(adata, 
    #                                       alpha_array=alpha_list, 
    #                                       batch_size=60)

    #     # Checking that optimal alpha is what we expect
    #     self.assertEqual(alpha_star, 0.1, 
    #                      ("scmkl.optimize_alpha chose the wrong alpha "
    #                       "as optimal for unimodal"))
        

    # def test_multimodal_optimize_alpha(self):
    #     """
    #     This function ensure that scmkl.optimize_alpha works correctly 
    #     for multimodal test cases by checking that the output is what 
    #     we expect.
    #     """
    #     adatas = [create_test_adata('RNA'), create_test_adata('ATAC')]
    #     alpha_list = np.array([0.01, 0.05, 0.1])

    #     alpha_star = scmkl.optimize_alpha(adatas, alpha_array=alpha_list,
    #                                       batch_size=60)

    #     self.assertEqual(alpha_star, 0.1, 
    #                      ("scmkl.optimize_alpha chose the wrong alpha "
    #                       "as optimal for multimodal run"))

    
    # def test_multiclass_optimize_alpha(self):
    #     """
    #     This function ensures finding the optimal alpha for best 
    #     performance via cross validation for multiclass runs is working 
    #     properly.
    #     """
    #     adata = read_h5ad('RNA')
    #     alphas = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    #     train_dict = scmkl.get_class_train(adata.uns['train_indices'],
    #                                        adata.obs['labels'], 
    #                                        adata.uns['seed_obj'])

    #     alpha_dict = scmkl.optimize_alpha(adata, 
    #                                       alpha_array=alphas, 
    #                                       batch_size=26, 
    #                                       train_dict=train_dict,
    #                                       metric='F1-Score')

    #     expected_alphas = {
    #         'B': 0.6, 
    #         'CD14+ Monocytes': 0.4, 
    #         'CD16+ Monocytes': 0.6, 
    #         'CD4 T': 0.3, 
    #         'CD8 T': 0.3, 
    #         'Dendritic': 0.5, 
    #         'NK': 0.5}

    #     self.assertDictEqual(alpha_dict, expected_alphas, 
    #                          ("scmkl.optimize_alpha() for multiclass "
    #                           "runs returned incorrect values."))
        

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
            'B': 0.8, 
            'CD14+ Monocytes': 0.7, 
            'CD16+ Monocytes': 0.6, 
            'CD4 T': 0.8, 
            'CD8 T': 0.8, 
            'Dendritic': 0.8, 
            'NK': 0.7}
        
        self.assertDictEqual(alpha_dict, expected_alphas, 
                             ("scmkl.optimize_alpha() for multiclass "
                              "runs returned incorrect values."))
        

if __name__ == '__main__':
    unittest.main()