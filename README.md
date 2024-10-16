<img src="scMKL_logo.png" alt="drawing" width="400"/>

This is an introduction to single cell Multiple Kernel Learning (scMKL). scMKL is a classification algorithm utilizing prior information to group features to enhance classification and aid understanding of distinguishing features in multi-omic data sets.


## Installation
1) Create an environment with scMKL and activate:
    ```bash
    # Create a conda env with python=3.11 and celer=0.7.3
    conda create -n scMKL python=3.11.2 conda-forge::celer=0.7.3
    conda activate scMKL
    ```
2) Clone scMKL reposity:
    ```bash
    git clone https://github.com/ohsu-cedar-comp-hub/scMKL.git
    ```
3) Install scMKL distribution with pip:
    ```bash
    # Navigate to dist directory in repo
    cd scMKL/dist/
    pip install scmkl-0.1.1.tar.gz
    ```

## Usage
scMKL takes advantage of AnnData objects and can be implemented with just four pieces of data:
1) scRNA and/or scATAC matrices (can be a csc_matrix)
2) An array of cell labels
3) An array of feature names (eg. gene symbols for RNA or peaks for ATAC)
4) A grouping dictionary where {'group_1' : [feature_5, feature_16], 'group_2' : [feature_1, feature_4, feature_9]}

For more information on formatting/creating the grouping dictionaries, see our example for creating an [RNA grouping](example/getting_RNA_groupings.ipynb) or [ATAC grouping](example/getting_ATAC_groupings.ipynb).

For implementing scMKL, see our examples for your use case in [examples](./example/).


## Citation
If you use scMKL in your research, please cite using:
```
To be determined
```
