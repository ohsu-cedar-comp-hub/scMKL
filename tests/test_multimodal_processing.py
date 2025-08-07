import unittest
import scmkl
import numpy as np
from test_calculate_z import create_test_adata


class TestMultimodalProcessing(unittest.TestCase):
    """
    This unittest class is used to ensure 
    scmkl.multimodal_processing() is working properly.
    """

    def test_multimodal_processing(self):
        """
        To test whether or not scmkl.multimodal_processing() is 
        working as expected, two adatas will be created and processed. 
        Then, the new dictionary names will be check along with the 
        length of group dict. Then, the dimensions of the resulting 
        adata is checked to ensure the correct number of samples and 
        combined features are present.  
        """
        # Creating adatas
        rna_adata = create_test_adata('RNA')
        atac_adata = create_test_adata('ATAC')

        # Combining adatas
        adatas = [rna_adata, atac_adata]
        names = ['RNA', 'ATAC']
        tfidf_list = [False, False]

        # Combining adatas
        adata = scmkl.multimodal_processing(adatas, names, 
                                            tfidf_list, 
                                            batch_size=80)

        # Capturing prepended names in group_dict
        prep_names = {gname.split('-')[0] 
                      for gname in adata.uns['group_dict'].keys()}
        
        # Ensuring new group names created properly
        self.assertSetEqual(set(names), prep_names, 
                            "Group names incorrectly modified")

        # Ensuring number of groups in correct
        self.assertEqual(len(adata.uns['group_dict']), 100, 
                         "Incorrect number of groupings in combined adata")
        
        # Ensuring number of samples remains the same
        self.assertEqual(adata.shape[0], 1000, 
                         "Incorrect number of samples in final adata")
        
        # Ensuring features were combined correctly
        self.assertEqual(adata.shape[1], 73202, 
                         "Incorrect number of features in final adata")
        

if __name__ == '__main__':
    unittest.main()