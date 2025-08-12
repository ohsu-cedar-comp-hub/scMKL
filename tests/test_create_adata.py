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
        err_str = ("Genes present in 'adata' group_dict "
                           "not in original grouping.")
        for group in adata.uns['group_dict'].keys():
            for gene in adata.uns['group_dict'][group]:
                
                self.assertIn(gene, grouping[group], err_str)
        
        # Checking that the number of dimensions for n = 1000 is correct
        self.assertEqual(adata.uns['D'], 100, "Incorrect optimal D calculated")

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


    def test_obs_retention(self):
        """
        This function is to ensure that `scmkl.create_adata()` is 
        retaining the correct cell labels despite resorting.

        This entails:
            - Making sure cells are sorted as all training, then all 
            testing.
            - Cell names still correspond to the correct row.
            - The training and test indices have been changed to 
            reflect the newly sorted `adata`. 
        """
        # Creating 10x10 matrix where every element in each row is the rows 
        # index (ex. row 5 is 10 5s)
        x = np.array([
            [i for j in range(10)]
            for i in range(10)
        ])
        obs = np.array([f'cell_{i}' for i in range(10)])
        var = np.array([f'feature_{i}' for i in range(10)])
        tt = ['train', 'test']
        train_test = np.array([tt[i%2] for i in range(10)])
        grouping = {'group1' : ['feature_1, feature_2'],
                    'group2' : ['feature_3', 'feature_4']}

        adata = scmkl.create_adata(x, var, train_test, grouping, obs, 
                                   split_data=train_test, 
                                   remove_features=False)
        
        # Making sure order matches obs_names
        # Should be all evens then all odds
        exp_order = ([i for i in range(10) if i%2 == 0] 
                  + [i for i in range(10) if i%2 == 1])
        exp_obs = [f'cell_{i}' for i in exp_order]

        exp_train = np.arange(0, 5)
        exp_test = np.arange(5, 10)
        elements_1d = [adata.X[i][0] for i in range(10)]

        eq_obs = np.all(adata.obs_names == exp_obs)
        eq_train = np.all(adata.uns['train_indices'] == exp_train)
        eq_test = np.all(adata.uns['test_indices'] == exp_test)
        matched_label = np.all(exp_order == elements_1d)

        self.assertTrue(eq_obs, "`.obs_names` are not in the correct order.")
        self.assertTrue(eq_train, "Train indices are not as expected.")
        self.assertTrue(eq_test, "Test indices are not as expected.")
        self.assertTrue(matched_label, 
                        "Elements in `adata.X` no longer match `.obs_names`")


    def test_format_adata(self):
        """
        This function ensures that format adata is working correctly 
        by comparing it's output to `create_adata()`.
        """
        # Read-in data
        x, grouping, features, labels = read_data('RNA')
        train = ['train'] * 800
        test = ['test'] * 200
        train_test = np.array(train + test)
        obs = np.array([f'cell_{i}' for i in range(x.shape[0])])

        # Creating adata to test, remove features must be `False` as it 
        # unreliably resorts features
        adata = scmkl.create_adata(X = x, feature_names = features, 
                                   obs_names=obs,
                                   cell_labels = labels, 
                                   group_dict = grouping,
                                   remove_features=False, 
                                   split_data=train_test)
        
        new_adata = ad.AnnData(X=x)
        new_adata.var_names = features
        new_adata.obs_names = obs
        new_adata.obs['treatment'] = labels
        new_adata = scmkl.format_adata(new_adata, 'treatment', grouping,
                                       remove_features=False, split_data=train_test)

        # Checking that adatas are the same
        expected_keys = {'group_dict', 'seed_obj', 'scale_data', 'D', 
                         'kernel_type', 'distance_metric', 'reduction', 
                         'tfidf', 'labeled_test', 'train_indices', 
                         'test_indices'}
        
        for key in expected_keys:
            self.assertIn(key, new_adata.uns.keys(), 
                          f"{key} is not in `new_adata`.")
            
        eq_mats = np.all(adata.X.toarray() == new_adata.X.toarray())
        eq_var_names = np.all(adata.var_names == new_adata.var_names)
        eq_obs_names = np.all(adata.obs_names == new_adata.obs_names)
        eq_labs = np.all(adata.obs['labels'] == new_adata.obs['labels'])

        self.assertTrue(eq_mats, 
                         "`.X` does not match between adatas.")
        self.assertTrue(eq_var_names,
                         "`.var_names` do not match between adatas.")
        self.assertTrue(eq_obs_names, 
                        "`.obs_names` do not match between adatas.")
        self.assertTrue(eq_labs, 
                        "`.obs['labels']` do not match between adatas.")
        self.assertEqual(adata.uns['D'], new_adata.uns['D'],
                         "`.uns['D']` does not match between adatas.")
        self.assertEqual(adata.uns['kernel_type'], 
                         new_adata.uns['kernel_type'],
                         ("`.uns['kernel_type']` does not match between "
                          "adatas."))
        self.assertEqual(adata.uns['distance_metric'], 
                         new_adata.uns['distance_metric'],
                         ("`.uns['distance_metric']` does not match between "
                          "adatas."))

if __name__ == '__main__':
    unittest.main()