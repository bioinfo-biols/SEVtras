Troubleshooting
----------------------

1. What do the numbers printed during running SEVtras mean, e.g. "0 30"?

 The first number represents the number of EM rounds in the current iteration, and the second represents the number of enriched representative genes in the current iteration.

2. What is the meaning of "sEV is hard to detect in xxx"? 

 The output means that SEVtras cannot detect a strong sEV signal in your sample. But if you really want to characterize potential sEVs, you can try to set the parameter "score_t" lower, e.g. '10'. (Note: this parameter must be a string.)

3. Error with `ValueError: max() arg is an empty sequence`?

 This means that SEVtras cannot find a representative gene to identify sEVs. There are two options you can choose. One is that you can try to lower the parameter "alpha", e.g. 0.09. The other is that you can use more samples to get a comprehensive result of representative genes.

4. How to extract the information that the vesicle belongs to which cell secretes it? 
   
 You can find this in the obsm of "SEVtras_sEVs.h5ad" indexed as source. Little reminder: this information can only be used as a clue and must be corroborated by other evidences.

5. Error with `ValueError: Length mismatch in deconvolver`? 
   
 This error may be caused by the dimension of the input cell matrix not matching the SEV matrix in some samples. 
 SEVtras deconvolver needs the corresponding sample information in the input cell matrix for each sEV. 
 
 Thus, please check that the sample name in the sEV matrix matches the sample name in the cell matrix (this should be contained in adata.obs, default is key 'batch' in the sEV matrix and cell matrix).

6. Why does the output file only contain "raw_Sample.h5ad" when I input a single scRNA-seq sample for SEVtras? 
   
 SEVtras is a data-driven algorithm to identify sEV-droplets. The reliability of the identification results increases with the number of inputted samples. It is recommended to input more than one sample for more reliable results. If the necessary data is not available, two copies of the same sample can be created and SEVtras will generate the corresponding results.

7. Does SEVtras support h5ad file converted from Seurat? 

 Yes. There is a `tutorial <https://www.youtube.com/watch?v=-MATf22tcak>`_ to convert Seurat to h5ad. I copied as following: 

 .. code-block:: R
 
    # load libraries 
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

    # write dimensional reduction matrix (PCA)
    write.csv (NML@reductions$pca@cell.embeddings, 
            file='../Seurat to H5AD/pca.csv', quote=F, row.names=F)

    # write gene names
    write.table(data.frame('gene'=rownames(counts_matrix)),
                file='../Seurat to H5AD/gene_names.csv',
                quote=F,row.names=F,col.names=F)

    view(NML@meta.data)

    # save metadata table:
    NML$barcode <- colnames(NML)
    NML$UMAP_1 <- NML@reductions$umap@cell.embeddings[,1]
    NML$UMAP_2 <- NML@reductions$umap@cell.embeddings[,2]
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


