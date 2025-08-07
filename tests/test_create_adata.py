import unittest
import scmkl
import anndata as ad
import numpy as np
from scipy.sparse import load_npz


def read_data(mod = 'RNA'):
    """
    Simple function to load example data to run tests.
    """
    x = load_npz(f'../example/data/_MCF7_{mod}_X.npz')
    grouping = np.load(f'../example/data/_{mod}_hallmark_groupings.pkl',
                            allow_pickle = True)
    features = np.load(f'../example/data/_MCF7_{mod}_feature_names.npy', 
                    allow_pickle = True)
    labels = np.load('../example/data/_MCF7_cell_labels.npy', 
                    allow_pickle = True)
    
    return x, grouping, features, labels


class TestCreateAdata(unittest.TestCase):
    """
    This unittest class is designed to test the scmkl.create_adata() 
    function. It creates an anndata.AnnData object and checks the 
    attributes required for scmkl to run.
    """
    def test_create_adata(self):
        """
        This function creates a scmkl AnnData object and checks the 
        train/test split, grouping dictionary, number of dimensions, 
        kernel function, and distance metric.
        """
        # Read-in data
        x, grouping, features, labels = read_data()
        train = ['train'] * 800
        test = ['test'] * 200
        train_test = np.array(train + test)
        d = scmkl.calculate_d(len(labels))

        # Creating adata to test
        adata = scmkl.create_adata(X = x, feature_names = features, 
                                   cell_labels = labels, D = d,
                                   group_dict = grouping,
                                   split_data = train_test)

        # Ensuring group dict is intact after object creation
        for group in adata.uns['group_dict'].keys():
            for gene in adata.uns['group_dict'][group]:
                err_str = (f"Genes present in 'adata' group_dict "
                           "not in original grouping.")
                self.assertIn(gene, grouping[group], err_str)
        
        # Checking that the number of dimensions for n = 1000 is correct
        self.assertEqual(adata.uns['D'], 61, "Incorrect optimal D calculated")

        # Checking default kernel function
        self.assertEqual(adata.uns['kernel_type'].lower(), 'gaussian', 
                         'Default kernel function should be gaussian')

        # Checking default distance metric
        self.assertEqual(adata.uns['distance_metric'], 'euclidean',
                         "Default distance metric should be euclidean")

        # Ensuring train test split is conserved when provided
        train_idx = np.where('train' == train_test)[0]
        train_bool = np.array_equal(adata.uns['train_indices'], train_idx)
        test_idx = np.where('test' == train_test)[0]
        test_bool = np.array_equal(adata.uns['test_indices'], test_idx)
        
        self.assertTrue(train_bool, "Train indices incorrect")
        self.assertTrue(test_bool, "Test indices incorrect")


if __name__ == '__main__':
    unittest.main()