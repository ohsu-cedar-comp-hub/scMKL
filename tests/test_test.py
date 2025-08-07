import unittest
import scmkl
import numpy as np
from test_train_model import train_scmkl


class TestTest(unittest.TestCase):
    """
    This unittest class is used to evaluate whether the functions in 
    scmkl.test are working properly.
    """

    def test_predict(self):
        """
        This function ensures that the output model from 
        scmkl.predict() returns as expected by checking the number of 
        coefficients.
        """
        # Creating adata
        adata = train_scmkl()

        metrics = ['AUROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']

        predictions, metrics_dict = scmkl.predict(adata, metrics)

        # Checking that metrics are present
        mets_check = [met in metrics_dict.keys() 
                      for met in metrics]
        
        self.assertTrue(np.all(mets_check), 
                        "There are missing metrics in metrics_dict")
        
        # Checking that output is binary
        self.assertTrue(np.unique(predictions).shape[0] == 2, 
                        "There are not only two classes in predictions")


    def test_find_selected_groups(self):
        """
        This function tests the scmkl.find_selected_groups() function 
        by ensuring the expected number of groups are output by the 
        function for the trained model.
        """
        # Training scMKL
        adata = train_scmkl()

        # Finding the selected groups from the model
        selected_groups = scmkl.find_selected_groups(adata)

        # Ensuring the number of output groups are expected
        self.assertEqual(len(selected_groups), 47,
                         "Incorrect number of selected groups")


if __name__ == '__main__':
    unittest.main()