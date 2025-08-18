"""
.. include:: ../README.md
.. include:: ../example/README.md

----------------------------

## **scMKL Documentation**
"""


__all__ = ['calculate_z', 
           'calculate_d',
           'create_adata', 
           'data_processing',
           'dataframes',
           'estimate_sigma', 
           'extract_results',
           'find_candidates',
           'format_adata',
           'format_group_names',
           'get_gene_groupings',
           'get_metrics',
           'get_region_groupings',
           'get_selection',
           'get_summary', 
           'get_weights', 
           'groups_per_alpha',
           'group_umap',
           'multimodal_processing', 
           'one_v_rest', 
           'optimize_alpha', 
           'optimize_sparsity',
           'parse_metrics',
           'parse_weights',
           'plotting',
           'plot_metric',
           'plot_conf_mat',
           'projections',
           'read_files',
           'read_gtf',
           'run',
           'sort_groups',
           'test',
           'tfidf_normalize',
           'train_model',
           'weights_barplot',
           'weights_dotplot',
           'weights_heatmap'
           ]

from scmkl._checks import *
from scmkl.calculate_z import *
from scmkl.create_adata import *
from scmkl.data_processing import *
from scmkl.dataframes import *
from scmkl.estimate_sigma import *
from scmkl.get_gene_groupings import *
from scmkl.get_region_groupings import *
from scmkl.multimodal_processing import *
from scmkl.one_v_rest import *
from scmkl.optimize_alpha import *
from scmkl.optimize_sparsity import *
from scmkl.plotting import *
from scmkl.projections import *
from scmkl.run import *
from scmkl.test import *
from scmkl.tfidf_normalize import *
from scmkl.train_model import *
from scmkl.projections import *