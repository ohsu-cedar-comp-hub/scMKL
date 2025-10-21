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
        # Creating adata and setting alphas to choose from 
        adata = create_test_adata()
        alpha_list = np.array([0.01, 0.05, 0.1])

        # Finding optimal alpha
        alpha_star = scmkl.optimize_alpha(adata, alpha_array = alpha_list, batch_size=60)

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
        # Creating two adatas
        rna_adata = create_test_adata('RNA')
        atac_adata = create_test_adata('ATAC')

        # Setting variables to run optimize_alpha
        adatas = [rna_adata, atac_adata]
        alpha_list = np.array([0.01, 0.05, 0.1])
        tfidf_list = [False, False]
        d = scmkl.calculate_d(rna_adata.shape[0])
        group_size = 2 * d

        # Finding optimal alpha
        alpha_star = scmkl.optimize_alpha(adatas, alpha_array = alpha_list, 
                                          group_size = group_size, 
                                          tfidf = tfidf_list, batch_size=60)

        # Checking that optimal_alpha is what we expect
        self.assertEqual(alpha_star, 0.1, 
                         ("scmkl.optimize_alpha chose the wrong alpha "
                          "as optimal for multimodal run"))

    
    def test_multiclass_optimize_alpha(self):
        """
        This function ensures finding the optimal alpha for best 
        performance via cross validation for multiclass runs is working 
        properly.
        """
        alphas = np.array([0.05, 0.1, 0.3, 0.5])
        adata = read_h5ad('RNA')

        alpha_dict = scmkl.optimize_alpha(adata, 
                                          alpha_array=alphas, 
                                          batch_size=26)

        expected_alphas = {
            'B': np.float64(0.5), 
            'CD14+ Monocytes': np.float64(0.5), 
            'CD16+ Monocytes': np.float64(0.5), 
            'CD4 T': np.float64(0.3), 
            'CD8 T': np.float64(0.3), 
            'Dendritic': np.float64(0.3), 
            'NK': np.float64(0.3)
        }

        self.assertDictEqual(alpha_dict, expected_alphas, 
                             ("scmkl.optimize_alpha() for multiclass "
                              "runs returned incorrect values."))
        

if __name__ == '__main__':
    unittest.main()