import scmkl
import unittest
import numpy as np

from test_calculate_z import create_test_adata


def get_results(opt_alpha=False):
    """
    Runs scmkl from start to finish to test dataframe functions.
    """
    alpha_array = np.array([0.01, 0.1])

    adata = create_test_adata()
    adata = scmkl.calculate_z(adata, batch_size=80)
    results = scmkl.run(adata, alpha_array)

    if opt_alpha:
        results['Alpha_star'] = 0.1

    return results


class TestDataframes(unittest.TestCase):
    """
    This unittest class is used to ensure that 
    the dataframes module is working properly.
    """
    def test_parse_metrics(self):
        """
        This function ensures that scmkl.parse_metrics() works 
        as expected.
        """
        results = get_results(opt_alpha=True)
        metrics = scmkl.parse_metrics(results, include_as=True)

        # Ensuring there are only five rows with Alpha Star == True
        num_opt = np.sum(metrics['Alpha Star'])
        self.assertEqual(num_opt, 5, "Wrong number of alpha_stars.")
        
        # Ensuring only the correct alphas are present
        self.assertSetEqual({0.1, 0.01}, set(metrics['Alpha']),
                            "Alphas should be 0.1 and 0.01.")
        

    def test_parse_weights(self):
        """
        This function ensures that scmkl.parse_weights() works 
        as expected.
        """
        results = get_results(opt_alpha=True)
        weights = scmkl.parse_weights(results, include_as=True)
        
        # Ensuring there are only five rows with Alpha Star == True
        num_opt = np.sum(weights['Alpha Star'])
        self.assertEqual(num_opt, 50, "Wrong number of alpha_stars.")
        
        # Ensuring only the correct alphas are present
        self.assertSetEqual({0.1, 0.01}, set(weights['Alpha']),
                            "Alphas should be 0.1 and 0.01.")


    def test_get_metrics(self):
        """
        This function ensures that scmkl.get_metrics() works 
        as expected.
        """
        # Creating results to run in normal mode
        results = get_results(opt_alpha=True)
        metrics = scmkl.get_metrics(results, include_as=True)

        # Ensuring there are only five rows with Alpha Star == True
        num_opt = np.sum(metrics['Alpha Star'])
        self.assertEqual(num_opt, 5, "Wrong number of alpha_stars.")
        
        # Ensuring only the correct alphas are present
        self.assertSetEqual({0.1, 0.01}, set(metrics['Alpha']),
                            "Alphas should be 0.1 and 0.01.")

        # Creating results to run with rfiles
        rfiles = {'rep_1' : results, 'rep_2' : results}
        metrics = scmkl.get_metrics(rfiles=rfiles, include_as=True)
        
        # Ensuring there are only five rows with Alpha Star == True
        num_opt = np.sum(metrics['Alpha Star'])
        self.assertEqual(num_opt, 10, "Wrong number of alpha_stars.")
        
        # Ensuring only the correct alphas are present
        self.assertSetEqual({0.1, 0.01}, set(metrics['Alpha']),
                            "Alphas should be 0.1 and 0.01.")


    def test_get_weights(self):
        """
        This function ensures that scmkl.get_weights() works 
        as expected.
        """
        # Creating results to run in normal mode
        results = get_results(opt_alpha=True)
        weights = scmkl.get_weights(results, include_as=True)

        # Ensuring there are only five rows with Alpha Star == True
        num_opt = np.sum(weights['Alpha Star'])
        self.assertEqual(num_opt, 50, "Wrong number of alpha_stars.")
        
        # Ensuring only the correct alphas are present
        self.assertSetEqual({0.1, 0.01}, set(weights['Alpha']),
                            "Alphas should be 0.1 and 0.01.")

        # Creating results to run with rfiles
        rfiles = {'rep_1' : results, 'rep_2' : results}
        weights = scmkl.get_weights(rfiles=rfiles, include_as=True)
        
        # Ensuring there are only five rows with Alpha Star == True
        num_opt = np.sum(weights['Alpha Star'])
        self.assertEqual(num_opt, 100, "Wrong number of alpha_stars.")
        
        # Ensuring only the correct alphas are present
        self.assertSetEqual({0.1, 0.01}, set(weights['Alpha']),
                            "Alphas should be 0.1 and 0.01.")
        

    def test_get_selection(self):
        """
        This function ensures that scmkl.get_selection() works 
        as expected.
        """
        # Creating results to run in normal mode
        results = get_results(opt_alpha=True)
        
        # Creating results to run with rfiles
        rfiles = {'rep_1' : results, 'rep_2' : results}
        weights = scmkl.get_weights(rfiles=rfiles, include_as=True)
        
        selection = scmkl.get_selection(weights, order_groups=True)
        
        # Checking that each selection < number of reps
        cor_reps = selection['Selection'] <= len(rfiles)
        self.assertTrue(np.all(cor_reps), "Selection is greater than expect.")

        # Ensuring all groups are present per alpha
        num_rows = len(rfiles)*50
        self.assertEqual(num_rows, len(selection), 
                         "Incorrect number of rows in selection df.")


    def test_read_gtf(self):
        """
        Checks that scmkl.read_gtf() is functioning properly.
        """
        # Reading in gtf file
        gtf_fp = '../example/data/_hg38_subset_protein_coding.annotation.gtf'
        gtf = scmkl.read_gtf(gtf_fp, filter_to_coding=True)

        # Checking columns
        gtf_cols = set(gtf.columns)
        expected_cols = {'frame', 'attribute', 'end', 'strand', 'feature', 
                         'gene_name', 'start', 'source', 'score', 'chr'}
        self.assertSetEqual(gtf_cols, expected_cols, 
                            "Incorrect columns in gtf dataframe.")

        # Ensuring only protein coding genes are present
        is_protein_coding = ['protein_coding' in attr 
                             for attr in gtf['attribute']]
        self.assertEqual(np.sum(is_protein_coding), len(gtf),
                         "Nonprotein coding annotations present.")


if __name__ == '__main__':
    unittest.main()