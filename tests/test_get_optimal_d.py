import unittest
import scmkl
import numpy as np
import anndata as ad


def create_dummy_adata():
    """
    Creates a dummy unformatted adata object to test functions.
    """
    x = np.arange(60000).reshape(60000,1)
    x = np.concatenate([x, x, x], axis=1)
    vars = {f'gene_{i}' for i in range(x.shape[1])}
    labs = [
        'class_1', 'class_2', 'class_2', 
        'class_1', 'class_1', 'class_3'
        ]
    labs *= 10000

    group_dict = {
            'Group1': ['gene_0', 'gene_2'],
            'Group2': ['gene_0', 'gene_1']
        }

    adata = ad.AnnData(X=x)
    adata.obs['labels'] = labs
    adata.var_names = vars
    adata.uns['group_dict'] = group_dict
    adata.uns['train_indices'] = np.arange(0, 48000)
    adata.uns['test_indices'] = np.arange(48000, 60000)

    return adata


class TestGetOptimalD(unittest.TestCase):
    """
    This unittest class is used to evaluate whether scmkl.get_optimal_d 
    is working properly for binary and multiclass setups.
    """
    def test_calculate_d(self):
        """
        This funcation ensures that the calculate_d function is working 
        correctly.
        """
        msg = "D was incorrectly calculated."
        self.assertEqual(scmkl.calculate_d(10000), 222, msg)
        self.assertEqual(scmkl.calculate_d(1000), 100, msg)


    def test_get_median_size(self):
        """
        This function ensure that the median run size for multiclass 
        runs is working correctly.
        """
        adata = create_dummy_adata()
        balanced_d = scmkl.calculate_d(scmkl.get_median_size(adata, 1.5))

        self.assertEqual(balanced_d, 445, 
                         "Multiclass D calculation is incorrect.")
        

    def test_get_optimal_d(self):
        """
        This function simply tests that test_get_optimal_d works 
        correctly across multiple use cases.
        """
        adata = create_dummy_adata()
        bin_labs = np.array(['class_1', 'class_2']*30000)

        # Need to test D provided case
        formatted_adata = scmkl.format_adata(adata, bin_labs, 
                                             adata.uns['group_dict'], D=200)
        self.assertEqual(formatted_adata.uns['D'], 200, 
                         "Incorrect D when D provided.")
        
        # Need to test D not provided for binary case
        formatted_adata = scmkl.format_adata(adata, bin_labs, 
                                             adata.uns['group_dict'])
        self.assertEqual(formatted_adata.uns['D'], 587, 
                         "Incorrect D when D not provided for binary.")
        
        # Need to test D not provided for binary case
        formatted_adata = scmkl.format_adata(adata, 'labels', 
                                             adata.uns['group_dict'],
                                             allow_multiclass=True)
        self.assertEqual(formatted_adata.uns['D'], 445, 
                         "Incorrect D when D not provided for multiclass.")


if '__main__' == __name__:
    unittest.main()