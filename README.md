<h1 align="center">
<img src="https://github.com/ohsu-cedar-comp-hub/scMKL/blob/main/scMKL_logo.png?raw=true" width="500"/>
</h1><br>


![PyPI](https://img.shields.io/pypi/v/scmkl?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/scmkl)
[![Anaconda-Server Badge](https://anaconda.org/ivango17/scmkl/badges/version.svg)](https://anaconda.org/ivango17/scmkl)
[![Anaconda-Server Badge](https://anaconda.org/ivango17/scmkl/badges/downloads.svg)](https://anaconda.org/ivango17/scmkl)
[![Anaconda-Server Badge](https://anaconda.org/ivango17/scmkl/badges/latest_release_date.svg)](https://anaconda.org/ivango17/scmkl)


Single-cell analysis using Multiple Kernel Learning, scMKL, is a binary classification algorithm utilizing prior information to group features to enhance classification and aid understanding of distinguishing features in multi-omic data sets.


## Installation

### Conda install
Conda is the recommended method to install scMKL:

```
conda create -n scMKL python=3.12 
conda activate scMKL
conda install -c conda-forge ivango17::scmkl
```

### Pip install
First, create a virtual environment with `python>=3.11.1,<3.13`.

Then, install scMKL with:
```
# activate your new env with python>=3.11.1 and <3.13
pip install scmkl
```

If wheels do not build correctly, ensure ```gcc``` and ```g++``` are installed and up to date. They can be installed with ```sudo apt install gcc``` and ```sudo apt install g++```.

## Requirements
scMKL takes advantage of AnnData objects and can be implemented with just four pieces of data:

1) scRNA and/or scATAC matrices (can be `scipy.sparse` matrix)

2) An array of cell labels

3) An array of feature names (eg. gene symbols for RNA or peaks for ATAC)

4) A grouping dictionary where {'group_1' : [feature_5, feature_16], 'group_2' : [feature_1, feature_4, feature_9]}

For more information on formatting/creating the grouping dictionaries, see our example for creating an [RNA grouping](https://github.com/ohsu-cedar-comp-hub/scMKL/blob/main/example/getting_RNA_groupings.ipynb) or [ATAC grouping](https://github.com/ohsu-cedar-comp-hub/scMKL/blob/main/example/getting_ATAC_groupings.ipynb).

For implementing scMKL, see our examples for your use case in [examples](https://github.com/ohsu-cedar-comp-hub/scMKL/tree/main/example).


## Links
Repo: [https://github.com/ohsu-cedar-comp-hub/scMKL](https://github.com/ohsu-cedar-comp-hub/scMKL)

PyPI: [https://pypi.org/project/scmkl/](https://pypi.org/project/scmkl/)

Anaconda: [https://anaconda.org/ivango17/scmkl](https://anaconda.org/ivango17/scmkl)

API: [https://ohsu-cedar-comp-hub.github.io/scMKL/](https://ohsu-cedar-comp-hub.github.io/scMKL/)


## Citation
If you use scMKL in your research, please cite using:
```
To be determined
```
Our Shiny for Python application for viewing data produced from this work can be found here: [scMKL_analysis](https://huggingface.co/spaces/scMKL-team/scMKL_analysis)