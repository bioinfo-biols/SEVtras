Troubleshooting
----------------------

1. What do the numbers printed during running SEVtras mean, e.g. *"0 30"*?

 The first number represents the number of EM rounds in the current iteration, and the second represents the number of enriched representative genes in the current iteration.

2. What is the meaning of *"sEV is hard to detect in xxx"*? 

 The output means that SEVtras cannot detect a strong sEV signal in your sample. But if you really want to characterize potential sEVs, you can try to set the parameter "score_t" lower, e.g. '10'. (Note: this parameter must be a string.)

3. Error with ``ValueError: max() arg is an empty sequence``?

 This means that SEVtras cannot find a representative gene to identify sEVs. There are two options you can choose. One is that you can try to lower the parameter "alpha", e.g. 0.09. The other is that you can use more samples to get a comprehensive result of representative genes.

4. How to extract the information that the vesicle belongs to which cell secretes it? 
   
 You can find this in the obsm of "SEVtras_sEVs.h5ad" indexed as source. Little reminder: this information can only be used as a clue and must be corroborated by other evidences.

5. Error with ``ValueError: Length mismatch in deconvolver``? 
   
 This error may be caused by the dimension of the input cell matrix not matching the SEV matrix in some samples. 
 SEVtras deconvolver needs the corresponding sample information in the input cell matrix for each sEV. 
 
 Thus, please check that the sample name in the sEV matrix matches the sample name in the cell matrix (this should be contained in adata.obs, default is key 'batch' in the sEV matrix and cell matrix).

6. Memory overflow when running SEVtras? 
   
 The default parameter of SEVtras is to use almost all CPU cores in your Linux server. This can cause memory overflows if the Linux server has a lot of CPU cores. You can use fewer CPU cores with `predefine_threads=20` to solve this problem.

7. The ``OBSsample`` (sample index) should be the same in adata_ev and adata_cell, default is 'batch'. This can be 'orig.ident' in **Seurat**, you can unify it with `adata.obs['batch'] = adata.obs['orig.ident']`. 

8. The SEVtras process was interrupted after running a part of the samples. How can I continue the process?

 You only need run SEVtras for the remaining samples. After finished, you can used following code to integrate all samples: 

 .. code-block:: python 

    import SEVtras
    from SEVtras.main import sEV_aggregator
    sEV_aggregator(out_path='the path you set to output in the previous step', name_list=['the sample name1 in your list', 'the sample name2 in your list', 'the sample nameN in your list'], max_M=1000, score_t=1e-15, threads=30, search_UMI=500, flag=0)

9. The cell data was preprocessed by **Scanpy** / **Seurat**, how to input for SEVtras? When to use ``Xraw``?
    
 It is OK to pre-process your cell data, including normalization, filtering and batch correction, before entering them into SEVtras. You can input unprocessed cell data to SEVtras with the cell type information generated from the pre-processed one. After cell type assignment, please run the following command to make is categorical with `adata.obs['celltype'] = pd.Categorical(adata.obs['celltype_clusters'].astype(str))`. 

 The reason for setting ``Xraw=True`` is to ensure that sEV-characterized genes are not filtered out in the pre-processing and filtering steps of the cell matrix. If you are able to save all gene expressions of the cell matrix during the conversion, I encourage you to save them and set ``Xraw=True``. If the raw data cannot be saved in some cases, you can follow the solution above. 

 Noted: ``Xraw`` does not mean the *raw_feature_bc_matrix* in Cell Ranger outs, it means the unprocessed cell data without filtering steps.

10. Does SEVtras only apply to humans and mice, and not to other non-model species? 
    
 The sEV-characteristic gene set is not sufficient for other non-model species. Currently, SEVtras only supports human and mouse. 

10. Why does the output file only contain "raw_Sample.h5ad" when I input a **single** scRNA-seq sample for SEVtras? 
   
 SEVtras is a data-driven algorithm to identify sEV-droplets. The reliability of the identification results increases with the number of inputted samples. It is recommended to input more than one sample for more reliable results. If the necessary data is not available, two copies of the same sample can be created and SEVtras will generate the corresponding results.

10. Does SEVtras support single nucleus RNA-seq data data? 
   
 SEVtras doesn't support single nucleus RNA-seq data (snRNA-seq). This is due to the experimental procedure of snRNA-seq. Small extracellular vesicles (sEVs) would be almost filtered out during the nuclei isolation and extraction process, so our software is not suitable for analyzing this type of data. 

11. Does SEVtras support h5ad file converted from **Seurat**? 

 Yes. There is a `tutorial <https://www.youtube.com/watch?v=-MATf22tcak>`_ to convert Seurat to h5ad. I copied as following: 
 
 ## Save in R 

 .. code-block:: R

    # load libraries 
    library(Seurat) 
    library(tidyverse) 
    library(Matrix) 
    # Read RDS created in Seurat Video Tutorials--video 7
    NML <- readRDS("../Seurat to H5AD/MergedNML.integrated.RDS")
    DimPlot(NML, reduction = "umap", label = TRUE)

    # write matrix data (gene expression counts) 
    counts_matrix <- GetAssayData(NML, assay='RNA', slot='counts')
    writeMM(counts_matrix, file=paste0(file='../Seurat to H5AD/matrix.mtx'))

    # write gene names
    write.table(data.frame('gene'=rownames(counts_matrix)),
                file='../Seurat to H5AD/gene_names.csv',
                quote=F,row.names=F,col.names=F)

    view(NML@meta.data)

    ## optional, if exists, then run
    # write dimensional reduction matrix (PCA)
    write.csv (NML@reductions$pca@cell.embeddings, 
            file='../Seurat to H5AD/pca.csv', quote=F, row.names=F)
    
    # save UMAP
    NML$UMAP_1 <- NML@reductions$umap@cell.embeddings[,1]
    NML$UMAP_2 <- NML@reductions$umap@cell.embeddings[,2]
    ## 
    
    # must run
    # save metadata table:
    NML$barcode <- colnames(NML)
    write.csv(NML@meta.data, file='../Seurat to H5AD/metadata.csv', 
            quote=F, row.names=F)


 ## Read by python 
 
 .. code-block:: python

    import scanpy as sc
    import anndata
    from scipy import io
    from scipy.sparse import coo_matrix, csr_matrix
    import os

    X = io.mmread("../Seurat to H5AD/matrix.mtx")
    adata = anndata.AnnData(X=X.transpose().tocsr())
    metadata = pd.read_csv("../Seurat to H5AD/metadata.mtx")

    with open("../Seurat to HSAD/gene_names.csv",'r') as f:
        gene_names = f.read().splitlines()

    adata.obs = metadata
    adata.obs.index = adata.obs["barcode"]
    adata.var.index = gene_names

    adata.write_h5ad('../Seurat to H5AD/adata.h5ad')

 After conversion, you also need to set the cell cluster in adata.obs to Categorical with `adata.obs['celltype'] = pd.Categorical(adata.obs['seurat_clusters'].astype(str))`.

