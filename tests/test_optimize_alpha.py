import unittest
import scmkl
import numpy as np
from test_calculate_z import create_test_adata


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

if __name__ == '__main__':
    unittest.main()