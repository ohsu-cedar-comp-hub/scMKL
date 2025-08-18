import scmkl
import unittest
import numpy as np


class TestGetGeneGrouping(unittest.TestCase):
    """
    This unittest class is used to ensure that 
    scmkl.get_gene_groupings() works properly.
    """

    def test_find_candidates(self):
        """
        This function runs and test the scmkl.find_candidates() 
        function.
        """
        candidates = scmkl.find_candidates('human', key_terms=[' b ', ' t '])

        self.assertEqual(candidates.shape[0], 12, 
                         "Incorrect number of rows, should be 12.")
        self.assertEqual(candidates.shape[1], 3, 
                         "Incorrect number of cols, should be 3.")
        
        no_matching = {39, 55, 0, 161, 115, 100, 2, 1, 10}
        res_matching = set(candidates['No. Key Terms Matching'])

        self.assertSetEqual(no_matching, res_matching, 
                            "Incorrect values in Key term matching column.")


    def test_get_gene_groupings(self):
        """
        This function runs and test the scmkl.get_gene_groupings() 
        function.
        """
        features = np.load('../example/data/_MCF7_RNA_feature_names.npy', 
                            allow_pickle=True)
        
        group_dict = scmkl.get_gene_groupings('CellMarker_2024', 'human',
                                                key_terms=[' b ', ' t '],
                                                min_overlap=2, 
                                                genes=features)
        self.assertEqual(len(group_dict), 161,
                         "Incorrect number of groupings kept w/o blacklist.")
        
        group_dict = scmkl.get_gene_groupings('CellMarker_2024', 'human',
                                                key_terms=[' b ', ' t '],
                                                min_overlap=2, 
                                                blacklist=['blood', 'stomach'],
                                                genes=features)
        self.assertEqual(len(group_dict), 99,
                         "Incorrect number of groupings kept w/ blacklist.")

    
if __name__ == '__main__':
    unittest.main()