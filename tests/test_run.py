import unittest
import scmkl
import numpy as np
from test_calculate_z import create_test_adata


class TestRun(unittest.TestCase):
    """
    This unittest class is used to ensure scmkl.run() works for both 
    unimodal and multimodal experiments.
    """

    def test_unimodal_run(self):
        """
        This functions ensures that unimodal tasks will run properly 
        for scmkl.run() by checking the output for the correct keys, 
        that all alphas are accounted for, and all samples are 
        accounted for in each model.
        """
        # Creating adata
        adata = create_test_adata()

        # Estimating sigma and Z matrices
        adata = scmkl.estimate_sigma(adata, batch_size=80)
        adata = scmkl.calculate_z(adata, batch_size=80)

        # Running scMKL
        alpha_list = np.array([0.1, 0.25, 0.4])
        num_alphas = alpha_list.shape[0]
        results = scmkl.run(adata, alpha_list)

        expected_keys = ['Metrics', 'Selected_groups', 'Norms', 
                         'Predictions', 'Observed', 'Test_indices', 
                         'Group_names', 'Models', 'Train_time', 'RAM_usage', 
                         'Probabilities']
        
        # Adding training and testing n
        num_observed = len(results['Observed'])
        num_predicted = [results['Predictions'][alpha].shape[0] 
                         for alpha in results['Predictions'].keys()]
        num_predicted.append(num_observed)
        all_n = np.array(num_predicted, dtype = int)

        # Ensuring all samples are accounted for
        self.assertTrue(np.all(all_n == 200),
                         ("Samples from results are missing from 'Observed' "
                          "or 'Predicted'"))

        # Checking that all required keys are present in results
        self.assertTrue(np.all([name in results.keys() 
                                for name in expected_keys]), 
                        "results of scmkl.run() missing keys")
        
        # Checking that data for all alpha is captured
        self.assertEqual(num_alphas, len(results['Metrics'].keys()),
                         "Alphas are missing in the 'Metrics' dictionary")
        self.assertEqual(num_alphas, len(results['Norms'].keys()),
                         "Alphas are missing in the 'Norms' dictionary")
        self.assertEqual(num_alphas, len(results['Selected_groups'].keys()),
                         ("Alphas are missing in the 'Selected_groups' "
                          "dictionary"))
        self.assertEqual(num_alphas, len(results['Models'].keys()),
                         "Alphas are missing in the 'Models' dictionary")
        
    def test_multimodal_run(self):
        """
        This functions ensures that multimodal tasks will run properly 
        for scmkl.run() by checking the output for the correct keys, 
        that all alphas are accounted for, and all samples are 
        accounted for in each model.
        """
        rna_adata = create_test_adata('RNA')
        atac_adata = create_test_adata('ATAC')

        # Combining adatas
        adatas = [rna_adata, atac_adata]
        names = ['RNA', 'ATAC']
        tfidf_list = [False, False]
        alpha_list = np.array([0.1, 0.25, 0.4])
        num_alphas = alpha_list.shape[0]

        # Combining adatas
        adata = scmkl.multimodal_processing(adatas, names, 
                                            tfidf_list, batch_size=80)

        results = scmkl.run(adata, alpha_list)

        expected_keys = ['Metrics', 'Selected_groups', 'Norms', 
                         'Predictions', 'Observed', 'Test_indices', 
                         'Group_names', 'Models', 'Train_time', 'RAM_usage', 
                         'Probabilities']
        
        # Adding training and testing n
        num_observed = len(results['Observed'])
        num_predicted = [results['Predictions'][alpha].shape[0] 
                         for alpha in results['Predictions'].keys()]
        num_predicted.append(num_observed)
        all_n = np.array(num_predicted, dtype = int)

        # Ensuring all samples are accounted for
        self.assertTrue(np.all(all_n == 200),
                         ("Samples from results are missing from 'Observed' "
                          "or 'Predicted'"))

        # Checking that all required keys are present in results
        self.assertTrue(np.all([name in results.keys() 
                                for name in expected_keys]), 
                        "results of scmkl.run() missing keys")
        
        # Checking that data for all alpha is captured
        self.assertEqual(num_alphas, len(results['Metrics'].keys()),
                         "Alphas are missing in the 'Metrics' dictionary")
        self.assertEqual(num_alphas, len(results['Norms'].keys()),
                         "Alphas are missing in the 'Norms' dictionary")
        self.assertEqual(num_alphas, len(results['Selected_groups'].keys()),
                         ("Alphas are missing in the 'Selected_groups' "
                          "dictionary"))
        self.assertEqual(num_alphas, len(results['Models'].keys()),
                         "Alphas are missing in the 'Models' dictionary")
        

if __name__ == '__main__':
    unittest.main()