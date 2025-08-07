import unittest
import scmkl
import numpy as np
from test_calculate_z import create_test_adata


class TestEstimateSigma(unittest.TestCase):
    """
    This unittest class is used to test the scmkl.estimate_sigma() 
    and ensure that it is functioning properly.
    """
    
    def test_estimate_sigma(self):
        """
        To test if estimate sigma is functioning properly within 
        scmkl, sigmas are calculated using the example data and both 
        euclidean and cityblock distance metrics then checking that 
        they are all positive and the mean of the sigmas is what we 
        expect.
        """
        adata = create_test_adata()

        # Capturing euclidean sigmas
        adata = scmkl.estimate_sigma(adata, batch_size=80)
        euclidean_sigmas = adata.uns['sigma'].copy()

        # Capturing cityblock sigmas
        adata.uns['distance_metric'] = 'cityblock'
        adata = scmkl.estimate_sigma(adata, batch_size=80)
        cityblock_sigmas = adata.uns['sigma'].copy()

        # Ensuring all sigmas positive
        self.assertTrue(np.all(euclidean_sigmas > 0), 
                        "Euclidean sigmas are not positive")
        self.assertTrue(np.all(cityblock_sigmas > 0), 
                        "Cityblock sigmas are not postive")
        
        # Checking that each has expected mean values
        self.assertAlmostEqual(np.mean(euclidean_sigmas), 11.460746, 
                               places=4, msg=("Euclidean sigmas mean is "
                                                  "not expected value"))
        self.assertAlmostEqual(np.mean(cityblock_sigmas), 75.320433, 
                               places = 4, msg = ("Cityblock sigmas mean is "
                                                  "not expected value"))


if __name__ == '__main__':
    unittest.main()