<h1 align="center">
<img src="scMKL_logo.png" alt="drawing" width="400"/>
</h1><br>

Single-cell Multiple Kernel Learning, scMKL, is a classification algorithm utilizing prior information to group features to enhance classification and aid understanding of distinguishing features in multi-omic data sets.


## Installation with pip
For scMKL to install correctly, g++ must be installed.
```
sudo apt-get install g++
```

Installing scMKL into your environment:
```{bash}
# activate your new env
pip install scmkl
```

## Usage
scMKL takes advantage of AnnData objects and can be implemented with just four pieces of data:
1) scRNA and/or scATAC matrices (can be a `scipy.sparse` matrix)
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
