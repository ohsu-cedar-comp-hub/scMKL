import scmkl
import unittest
import numpy as np


class TestGetRegionGrouping(unittest.TestCase):
    """
    This unittest class is used to ensure that 
    scmkl.get_region_groupings() works properly.
    """

    def test_get_region_groupings(self):
        """
        This function ensures that scmkl.get_region_grouping() works 
        as expected.
        """
        # Reading in gtf file
        gtf_fp = '../example/data/_hg38_subset_protein_coding.annotation.gtf'
        gtf = scmkl.read_gtf(gtf_fp, filter_to_coding=True)
        
        # Reading in gene sets and feature names
        gene_sets = np.load('../example/data/_RNA_hallmark_groupings.pkl',
                            allow_pickle=True)
        regions = np.load('../example/data/_MCF7_ATAC_feature_names.npy', 
                          allow_pickle=True)
        
        # Creating epigenetic grouping, and empty grouping
        region_grouping = scmkl.get_region_groupings(gtf, gene_sets, regions)
        empty_grouping = scmkl.get_region_groupings(gtf, gene_sets, regions, 
                                                    len_up=10, len_down=10)

        # Reading in larger, complete peak set to check output
        truth_grouping = np.load('../example/data/_ATAC_hallmark_groupings.pkl', 
                                 allow_pickle=True)
        
        # Check output grouping
        self.assertEqual(len(region_grouping), 50, 
                         "Incorrect number of groupings, should be 50.")
        
        for group, regions in region_grouping.items():
            for region in regions:
                self.assertIn(region, truth_grouping[group], 
                              f"Feature {region} should not be in {group}")
                
        # Ensuring no peaks matched genes with such small promoter regions
        empty_sizes = [len(empty_grouping[group]) <= 2
                       for group in empty_grouping.keys()]
        self.assertTrue(np.all(empty_sizes), "Group sizes should be minimal.")                      


if __name__ == '__main__':
    unittest.main()