import unittest
import scmkl
import numpy as np
from test_create_adata import read_data


def create_test_adata(mod = 'RNA'):
    """
    Create a anndata.AnnData object with default params for scmkl
    to run for testing.
    """
    x, grouping, features, labels = read_data()
    d = scmkl.calculate_d(len(labels))

    adata = scmkl.create_adata(X = x, feature_names = features, 
                                   cell_labels = labels, D = d,
                                   group_dict = grouping, 
                                   remove_features = False)

    return adata


class TestCalculateZ(unittest.TestCase):
    """
    This unittest class is used to evaluate whether 
    scmkl.calculate_z() is functioning properly.
    """
    def test_calculate_z(self):
        """
        To check the output of calculate_z, z is calculated on both 
        training and testing data then dimensions are checked. Then 
        the maximum and minimum values are check to ensure that 1) 
        both Z_train and Z_test have the same distribution and the 
        values are what we would expect with the given params and data.
        """
        adata = create_test_adata()
        adata = scmkl.estimate_sigma(adata, batch_size=80)
        adata = scmkl.calculate_z(adata, batch_size=80)

        # Capturing theoretical dimensions of Z matrices
        n_cols = 2 * adata.uns['D'] * len(adata.uns['group_dict'])
        train_n_rows = adata.uns['train_indices'].shape[0]
        test_n_rows = adata.uns['test_indices'].shape[0]

        train_z_dims = (train_n_rows, n_cols)
        test_z_dims = (test_n_rows, n_cols)

        # Ensuring Z dims match what is expected
        self.assertEqual(adata.uns['Z_train'].shape, train_z_dims, 
                         "Z train dims are not correct")
        self.assertEqual(adata.uns['Z_test'].shape, test_z_dims,
                         "Z test dims are not correct")

        # Quick check on Z train distribution
        self.assertAlmostEqual(np.min(adata.uns['Z_train']), -0.12803687, 
                               places = 4, msg = ("Z_train min outside of "
                                                  "expected distribution"))
        self.assertAlmostEqual(np.max(adata.uns['Z_train']), 0.12803687, 
                               places = 4, msg = ("Z_train max outside of "
                                                  "expected distribution"))
        self.assertAlmostEqual(np.median(adata.uns['Z_train']), 0.08204988,
                               places = 4, msg = ("Z_train median is out of "
                                                  "bounds for expected dist"))

        # Quick check on Z test distribution
        self.assertAlmostEqual(np.min(adata.uns['Z_test']), -0.12803687, 
                               places = 4, msg = ("Z_test min outside of "
                                                  "expected distribution"))
        self.assertAlmostEqual(np.max(adata.uns['Z_test']), 0.12803687, 
                               places = 4, msg = ("Z_test max outside of "
                                                  "expected distribution"))
        self.assertAlmostEqual(np.median(adata.uns['Z_test']), 0.08180354,
                               places = 4, msg = ("Z_test median is out of "
                                                  "bounds for expected dist"))

if __name__ == '__main__':
    unittest.main()