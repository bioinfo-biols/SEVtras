Functions
-----------

The main functions in SEVtras are listed below:

.. code-block:: python

    SEVtras.sEV_recognizer(sample_file, out_path, input_path=None, species='Homo', predefine_threads=-2, get_only=False, score_t=None, search_UMI=500, alpha=0.1, dir_origin=True)  

This function used for sEV recognizing. 

* *sample_file*\: the path of each sample row by row,  
* *out_path*\: the path for output files, 
* *input_path*\: if all input files in the same directory, we can use this to represent the path, default is ``None``\, 
* *species*\: the species from which the scRNA-seq sample was sequenced, default is ``Homo``\. For mouse samples, you can use ``Mus``\, 
* *predefine_threads*\: SEVtras uses parallel processing for acceleration, we can define how many cpu cores to use, default is all cpu cores minus two ``-2``\,  
* *get_only*\: whether to read protein information in the adata, default is ``False``\,  
* *score_t*\: the threshold for SEVtras score to recognize sEVs, default is ``None``\. If no sEVs found in the *sEVs_SEVtras.h5ad*\, we can change *score_t* to a smaller threshold (``str``), e.g. '10', 
* *search_UMI*\: the UMI range to search for sEVs, default is ``500``, you can use ``200`` for stricter recognization,
* *alpha*\: the parameter for identifying sEV representative genes for each sample, default is ``0.10``. If you cannot detect sEVs in all samples, the parameter can be loosened to a smaller value, e.g. ``0.09``\,
* *dir_origin*\: the path of *matrix.mtx.gz*\, default is ``True`` assuming that the file of *matrix.mtx.gz* locates at ``sample/outs/raw_feature_bc_matrix/``\. If you set it as ``False``, SEVtras search for *matrix.mtx.gz* in the path of each sample in the parameter *sample_file*\. 

.. code-block:: python

    SEVtras.ESAI_calculator(adata_ev, adata_cell, out_path, species='Homo', OBSsample='batch', OBScelltype='celltype', OBSev='sEV', OBSMpca='X_pca', cellN=10, Xraw = True, normalW=True, plot_cmp='SEV_builtin', save_plot_prefix='', OBSMumap='X_umap',size=10) 

This function used for ESAI calculating. 

* *adata_ev*\: the path to sEV-anndata objects, 
* *adata_cell*\: the path to cell-anndata objects, 
* *out_path*\: the path for output files, 
* *species*\: the species from which the scRNA-seq sample was sequenced, default is ``Homo``\. For mouse samples, you can use ``Mus``\, 
* *OBSsample*\: the index represents the sample information in the ``obs`` of adata, default is ``batch``\, 
* *OBScelltype*\: the index represents the cell type information in the ``obs`` of adata, default is ``celltype``\, 
* *OBSev*\: the index represents the sEV information in the ``obs`` of adata, default is ``sEV``\, 
* *OBSMpca*\: the index represents the PCA information in the ``obsm`` of adata, default is ``X_pca``\, 
* *cellN*\: the number of neighors used for ESAI deconvolution, default is ``10``\, 
* *Xraw*\:  whether or not to use the raw object in the ``adata_cell``. If ``adata_cell`` has been filtered or normalized, please set ``Xraw=True``, and ``adata_cell.raw`` will be used. Note: if set ``Xraw=True``, save raw ``adata_cell`` as ``adata_cell.raw`` before filtering. Default is ``True``\, 
* *normalW*\: whether or not to scale ``adata_cell`` in ESAI deconvolution, default is ``True``\, 
* *plot_cmp*\: the pallete used for plot different cell types in umap, default is ``SEV_builtin``\, you can use other pallete in matplotlib e.g. ``Set2``\, 
* *save_plot_prefix*\: the prefix name for saved files, default is ``''``\, 
* *OBSMumap*\: the index represents the umap information in the ``obsm`` of adata, default is ``X_umap``\, 
* *size*\: the size of point in umap plot, default is ``10``\. 

.. code-block:: python

    SEVtras.sEV_imputation(adata_sEV) 

This function used for sEV data imputation.

.. code-block:: python

    SEVtras.cellfree_simulator(out_path, gene_exp_ev, gene_exp_cell, expect_UMI = [40, 70, 100, 130], sEV_fraction = [0.005, 0.01, 0.05, 0.10], sEV=500)

This function used for cell free droplets simulation. 

.. code-block:: python

    SEVtras.sEV_enrichment(adata_sEV, nBP=15) 

This function used for sEV data GO enrichment.

