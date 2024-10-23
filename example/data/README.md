## Data used in tutorials
To demonstrate the implementation of scMKL in our tutorials, we used a previously generated data set from Ors et al., 2022 subset to only 1,000 cells. Here is a summary of the included files:
| File | Description | Additional Information |
| ---- | ----------- | ---------------------- |
| MCF7_cell_labels.npy | Saved numpy array for MCF-7 cell labels | Respective to both RNA and ATAC MCF-7 data |
| MCF7_RNA_feature_names.npy | Saved numpy array for MCF-7 RNA features (gene symbols) | Respective to MCF7_RNA_X.npz |
| MCF7_RNA_X.npz | Saved scipy.sparse.csc_matrix of MCF-7 count matrix (cell x feature) | Respective to MCF7_RNA_feature_names.npy |
| MCF7_ATAC_feature_names.npy | Saved numpy array of feature names for MCF-7 ATAC features (regions) | Respective to MCF7_ATAC_X.npz |
| MCF7_ATAC_X.npz | Saved scipy.sparse.csc_matrix of MCF-7 binary ATAC matrix (cell x feature) | Respective to MCF7_ATAC_feature_names.npy |

Other files:
| File | Description | Additional Information |
| ---- | ----------- | ---------------------- |
| hallmark_library.gmt | A GMT file with Hallmark gene sets | Downloaded from [Molecular Signature Database](https://www.gsea-msigdb.org/gsea/msigdb) |
| hg38_subset_protein_coding.annotations.gtf | A 1,000 line GTF file containing only protein coding genes (hg38) | Downloaded from [Ensembl](https://useast.ensembl.org/index.html) | 
| RNA_hallmark_groupings.pkl | A pickle file of Hallmark gene sets | Created from GMT file downloaded from [Molecular Signature Database](https://www.gsea-msigdb.org/gsea/msigdb) |
| ATAC_hallmark_groupings.pkl | A pickle file of grouped genomic regions from Hallmark gene sets | Created by `scmkl.get_atac_groupings()` |