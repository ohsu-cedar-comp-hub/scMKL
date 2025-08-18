import scmkl
import os
import unittest
import numpy as np


def read_results(ftype='RNA', add_alpha_star=False):
    """
    Read scMKL results file.
    """
    if ftype != 'multiclass':
        file = f'../example/data/MCF7_RNA_results.pkl'
    else:
        file = '../example/data/celltype_results.pkl'

    file = np.load(file, allow_pickle=True)

    if add_alpha_star:
        file['Alpha_star'] = np.min(list(file['Norms'].keys()))

    return file


class TestPlotting(unittest.TestCase):
    """
    This unittest class is used to ensure that the plotting module is 
    running. Output plots for review will be placed in ./figures/.
    """
    def test_plot_conf_mat(self):
        """
        This function ensures that `scmkl.plot_conf_mat()` works 
        properly for both binary and multiclass results.
        """
        binary = read_results('RNA', True)
        multi = read_results('multiclass')

        scmkl.plot_conf_mat(binary, save='figures/plot_conf_mat_binary.png')
        scmkl.plot_conf_mat(multi, save='figures/plot_conf_mat_multi.png')


    def test_plot_metric(self):
        """
        This function ensures that `scmkl.plot_metric()` works 
        properly for both binary and multiclass results.
        """
        binary = read_results('RNA', True)
        multi = read_results('multiclass')

        binary_summary = scmkl.get_summary(binary)
        plot = scmkl.plot_metric(binary_summary, alpha_star=0.29)
        plot.save('figures/plot_metric_binary.png')

        multi_summary = scmkl.get_summary(multi)
        plot = scmkl.plot_metric(multi_summary)
        plot.save('figures/plot_metric_multi.png')

    
    def test_weights_barplot(self):
        """
        This function ensures that `scmkl.weights_barplot()` works 
        properly for both binary and multiclass results.
        """
        binary = read_results('RNA', True)
        multi = read_results('multiclass')

        binary_plot = scmkl.weights_barplot(binary, n_groups=50)
        binary_plot.save('figures/weights_barplot_binary.png')

        multi_plot = scmkl.weights_barplot(multi, n_groups=16)
        multi_plot.save('figures/weights_barplot_multi.png')


    def test_weights_heatmap(self):
        """
        This function ensures that `scmkl.weights_heatmap()` works 
        properly for both binary and multiclass results.
        """
        binary = read_results('RNA', True)
        multi = read_results('multiclass')

        binary_plot = scmkl.weights_heatmap(binary)
        binary_plot.save('figures/weights_heatmap_binary.png')

        multi_plot = scmkl.weights_heatmap(multi, scale_weights=True)
        multi_plot.save('figures/weights_heatmap_multi.png')


    def test_weights_dotplot(self):
        """
        This function ensures that `scmkl.weights_dotplot()` works 
        properly for both binary and multiclass results.
        """
        binary = read_results('RNA', True)
        multi = read_results('multiclass')

        binary_plot = scmkl.weights_dotplot(binary)
        binary_plot.save('figures/weights_dotplot_binary.png')

        multi_plot = scmkl.weights_heatmap(multi, scale_weights=True)
        multi_plot.save('figures/weights_dotplot_multi.png')


    def test_group_umap(self):
        """
        This function ensures that `scmkl.group_umap()` works 
        properly for both binary and multiclass results.
        """
        adata_fp = '../example/data/pbmc_{}.h5ad'
        group_fp = '../example/data/_{}_azimuth_pbmc_groupings.pkl'

        rna_adata = scmkl.format_adata(adata_fp.format('rna'), 
                                       'celltypes', 
                                       group_fp.format('RNA'), 
                                       allow_multiclass=True)
        scmkl.group_umap(rna_adata, g_name='B Cell Markers', 
                         title='B Cell Markers', save='_group_rna.png')
        
        atac_adata = scmkl.format_adata(adata_fp.format('atac'), 'celltypes',
                                        group_fp.format('ATAC'),
                                        allow_multiclass=True)

        scmkl.group_umap(atac_adata, g_name='CD16+ Monocyte Markers', 
                         is_binary=True, title='CD16+ Monocyte Markers', 
                         save='_group_atac.png')


if __name__ == '__main__':

    if not os.path.exists('figures/'):
        os.mkdir('figures/')

    unittest.main()