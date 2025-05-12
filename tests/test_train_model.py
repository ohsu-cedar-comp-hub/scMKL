import unittest
import scmkl
import numpy as np
from test_calculate_z import create_test_adata  


def train_scmkl():
    '''
    This function simply trains a small scMKL model to check the 
    outputs of scmkl.predict() and scmkl.find_selected_groups().
    '''
    adata = create_test_adata()
    group_size = adata.uns['D'] * 2
    alpha = 0.1

    adata = scmkl.estimate_sigma(adata)
    adata = scmkl.calculate_z(adata)
    adata = scmkl.train_model(adata, group_size, alpha)

    return adata


class TestTrainModel(unittest.TestCase):
    '''
    This unitest class is used to ensure that the scmkl.train_model() 
    function is working properly.
    '''

    def test_train_model(self):
        '''
        This function ensures that the expected values are returned by 
        scmkl.train_model() by checking the number of coefficients and 
        checking the min and max values of the coefficients.
        '''
        adata = train_scmkl()

        # Calculating group size
        group_size = adata.uns['D'] * 2

        # Calculating theoretical number of coefficients
        num_groups = len(adata.uns['group_dict'])
        cor_n_coefs = group_size * num_groups

        # Capturing number of coefficients
        coefs = adata.uns['model'].coef_
        print(np.max(coefs), np.min(coefs))
        
        # Ensuring that the GroupLasso model has the right number of coeffs
        self.assertEqual(len(coefs), cor_n_coefs, 
                         ("Generated GroupLasso model does not have the "
                          "correct number of coefficients"))
        
        # Checking that the min and max coeffs are expected values
        self.assertAlmostEqual(np.max(coefs), 0.85678, places = 4, 
                        msg = "Max coefficient is higher than expected")
        self.assertAlmostEqual(np.min(coefs), -0.70349, places = 4,
                        msg = "Min coefficient is lower than expected")
        

if __name__ == '__main__':
    unittest.main()