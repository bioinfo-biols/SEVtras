## SEVtras delineates small extracellular vesicles at droplet resolution from single-cell transcriptomes
SEVtras stands for <ins>sEV</ins>-con<ins>t</ins>aining d<ins>r</ins>oplet identific<ins>a</ins>tion in <ins>s</ins>cRNA-seq data.

You can freely use SEVtras to explore sEV heterogeneity at single droplet, characterize cell type dynamics in light of sEV activity and unlock diagnostic potential of sEVs in concert with cells.

<p align="center">
  <img src='./docs/SEVtras_overview.png'>
</p>
<p align="center">
  Overview of SEVtras.
</p>

### Prerequisites
    "numpy", "pandas", "scipy", "umap",
    "statsmodels", "gseapy", "scanpy"

### Installation
```bash
pip install SEVtras
```
We also suggest to use a separate conda environment for installing SEVtras.
```bash
conda create -y -n SEVtras_env python=3.7
source activate SEVtras_env
pip install SEVtras
```

### Simple Example
The pipeline of SEVtras only composed two parts: sEV_recognizer and ESAI_calculator. 

Part I:
```bash
SEVtras.sEV_recognizer(input_path='./tests', sample_file='./tests/sample_file', out_path='./outputs', species='Homo')
```

Part II:
```bash
SEVtras.ESAI_calculator(adata_ev_path='./outputs/sEV_SEVtras.h5ad', adata_cell_path='./tests/adata_cell.h5ad', out_path='./outputs', Xraw=False, OBSsample='batch', OBScelltype='celltype')
```

Further tutorials please refer to  https://SEVtras.readthedocs.io/.

### Citation
He, R., Zhu, J., Ji, P. et al. SEVtras delineates small extracellular vesicles at droplet resolution from single-cell transcriptomes. Nat Methods 21, 259-266 (2024). https://doi.org/10.1038/s41592-023-02117-1
