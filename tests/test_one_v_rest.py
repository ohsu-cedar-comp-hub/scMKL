import scmkl
import unittest
import numpy as np
from test_calculate_z import create_test_adata


class TestOneVRest(unittest.TestCase):
    """
    This unittest class is used to ensure that scmkl.one_v_rest() 
    works properly for both unimodal and multimodal runs.
    """

    def test_unimodal_one_v_rest(self):
        """
        This function ensure that scmkl.one_v_rest() works correctly 
        for unimodal test cases by checking that the output is what we 
        expect without failing.
        """
        
        # Creating adata and setting alphas to choose from
        adata = create_test_adata('RNA')
        alpha_list = np.array([0.1, 0.2])

        # Creating dummy labels and train/test split to test function
        ct = ['B', 'T', 'Mono']
        train_idc = np.arange(0, 800)
        test_idc = np.arange(800, 1000)

        adata.obs['labels'] = [ct[i%3] for i in range(adata.shape[0])]
        adata.uns['train_indices'] = train_idc
        adata.uns['test_indices'] = test_idc

        adata = scmkl.calculate_z(adata, batch_size=80)
        results = scmkl.one_v_rest([adata], ['rna'], 
                                   alpha_list=alpha_list, 
                                   tfidf=[False],
                                   batch_size=80)

        expected_keys = np.array(ct + ['Macro_F1-Score', 
                                       'Predicted_class', 
                                       'Truth_labels'])
        keys_in = np.array([key in results.keys() for key in expected_keys])
        
        self.assertTrue(np.all(keys_in), 
                        f"Expect keys missing: {expected_keys[~keys_in]}")

if __name__ == '__main__':
    unittest.main()