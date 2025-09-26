# scMKL Workshop, September 30th 2025

Ian VanGordon, Sam Kupp, and Cigdem Ak will be giving an scMKL workshop 
covering the following topics:

- How scMKL works
- Getting meaningful feature groupings for your dataset
- Running the model
- Interpreting results 


## Installing scMKL
### Conda install
Conda is the recommended method to install scMKL:

```bash
# Ensure conda-forge and bioconda channels are available
conda create -n scMKL python=3.12 
conda activate scMKL
conda install ivango17::scmkl

# install jupyter to run notebook
conda install -c conda-forge jupyter
```

### Pip install
First, create a virtual environment with `python>=3.11.1,<3.13`.

Then, install scMKL with:
```bash
# activate your new env with python>=3.11.1 and <3.13
pip install scmkl

# install jupyter to run notebook
pip install jupyter
```

If wheels do not build correctly, ensure ```gcc``` and ```g++``` are installed and up to date. They can be installed with ```sudo apt install gcc``` and ```sudo apt install g++```.


## Required files

If you have a `.h5ad` file with cell annotations, **ignore** the table below.

| Input Data | Compatable File Types |
| ---------- | --------------------- |
| ***X*** | `.npy`, `.npz`, `.csv`, `.tsv`,  `.mtx`, `.pkl` |
| Feature Names | `.npy`, `.csv`, `.tsv`, `.pkl` |
| Cell Annotations | `.npy`, `.csv`, `.tsv`,  `.pkl` |

If you do not have single-cell data that you want to run or is not formatted 
correctly, we have provided a single-cell multiome lymphoma (SLL) data set that 
you can run.
