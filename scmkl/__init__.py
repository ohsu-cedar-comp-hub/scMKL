"""
.. include:: ../README.md

----------------------------

## **scMKL Documentation**
"""


__all__ = ['calculate_z', 
           'create_adata', 
           'data_processing',
           'dataframes',
           'estimate_sigma', 
           'get_region_groupings',
           'multimodal_processing', 
           'one_v_rest', 
           'optimize_alpha', 
           'optimize_sparsity',
           'plotting',
           'projections',
           'read_gtf',
           'run',
           'test',
           'tfidf_normalize',
           'train_model'
           ]

from scmkl._checks import *
from scmkl.calculate_z import *
from scmkl.create_adata import *
from scmkl.data_processing import *
from scmkl.dataframes import *
from scmkl.estimate_sigma import *
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