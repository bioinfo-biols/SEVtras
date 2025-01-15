## source 
import gseapy as gp
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
import copy
import sys
import os
import pickle
# from pathlib import Path
from os import path

def source_biogenesis(adata_cell, species, OBScelltype='celltype', Xraw = True, normalW=True):
    if Xraw:
        X_input = adata_cell.raw
    else:
        X_input = adata_cell
        
    if normalW:
        try:
            import scanpy as sc
            X_norm = sc.pp.scale(X_input.X, zero_center=True, max_value=None, copy=False)
        except ImportError:
            from .sc_pp import scale
            X_norm = scale(X_input.X, zero_center=True, max_value=None, copy=False)
    else:
        X_norm = X_input.X
    
    gsea_pval = []
    num_clusters = len(adata_cell.obs[OBScelltype].cat.categories)

    if species == 'Homo':
        gmt_path = path.join(path.dirname(__file__), 'evs.gmt')#Path(__file__).parent / 'evs.gmt'
    elif species == 'Mus':
        gmt_path = path.join(path.dirname(__file__), 'evsM.gmt')

    for i in range(num_clusters):
        i = adata_cell.obs[OBScelltype].cat.categories[i]
        gene_rank = pd.DataFrame({'exp': np.array(X_norm[adata_cell.obs[OBScelltype] == str(i), :].mean(axis=0))}, index = X_input.var_names)

        res = gp.prerank(rnk=gene_rank, gene_sets=gmt_path)
        terms = res.results.keys()
        gsea_pval.append([i, res.results[list(terms)[0]]['nes'], res.results[list(terms)[0]]['pval']])        

    gsea_pval_dat = pd.DataFrame(gsea_pval, index=adata_cell.obs[OBScelltype].cat.categories, columns = ['num', 'enrich', 'p'])
    gsea_pval_dat['log1p'] = -np.log10(gsea_pval_dat['p']+1e-4) * np.sign(gsea_pval_dat['enrich'])

    return(gsea_pval_dat)


def near_neighbor(adata_combined, OBSsample='batch', OBSev='sEV', OBScelltype='celltype', OBSMpca='X_pca', cellN=10):
    ## run sc.pp.pca(adata_combined) before
    near_neighbor = []
    for sample in adata_combined.obs[OBSsample].unique().astype(str):
        tse_ref = copy.copy(adata_combined[(adata_combined.obs[OBSev] == '0') & (adata_combined.obs[OBSsample] == sample),])
        if tse_ref.shape[0] > 0:
            cell_tree = cKDTree(tse_ref.obsm[OBSMpca], leafsize=100)
            
            tmp_umap = adata_combined[(adata_combined.obs[OBSev] == '1') & (adata_combined.obs[OBSsample] == sample),].obsm[OBSMpca]
            for i in range(tmp_umap.shape[0]):
                TheResult = cell_tree.query(tmp_umap[i,], k=10)

                near_neighbor.append([sample, i] + list(tse_ref.obs[OBScelltype].iloc[TheResult[1]]))#tmp_umap.obs['clusters'][i]] +
        else:
            print('No matched cell sample for sEV deconvolution')

    near_neighbor_dat = pd.DataFrame(near_neighbor)
    return(near_neighbor_dat)

def preprocess_source(adata_ev, adata_cell, OBScelltype='celltype', OBSev='sEV', Xraw = True):
    ## cell type
    if Xraw:
        adata_cell_raw = copy.copy(adata_cell.raw.to_adata())
    else:
        adata_cell_raw = copy.copy(adata_cell)

    adata_ev.obs[OBScelltype] = OBSev
    adata_ev.obs[OBScelltype] = pd.Series(adata_ev.obs[OBScelltype], dtype="category")

    adata_combined = adata_cell_raw.concatenate(adata_ev, batch_key = OBSev)

    adata_combined.obs[OBScelltype] = pd.Categorical(adata_combined.obs[OBScelltype], \
        categories = np.append(adata_cell_raw.obs[OBScelltype].cat.categories.values, OBSev), ordered = False)
    
    adata_combined.raw = adata_combined
    try:
        import scanpy as sc
        sc.pp.normalize_total(adata_combined, target_sum=1e4)
        sc.pp.log1p(adata_combined)
        sc.pp.highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
        # sc.pl.highly_variable_genes(Normal_combined)
        adata_combined = adata_combined[:, adata_combined.var.highly_variable]#highly_variable

        sc.pp.pca(adata_combined)
        sc.pp.neighbors(adata_combined)
        sc.tl.umap(adata_combined)
    except ImportError:
        print('Scanpy is not installed. We use in-house script')
        from .sc_pp import normalize_total, log1p, highly_variable_genes, pca, neighbors
        from .sc_pp import umap
        normalize_total(adata_combined, target_sum=1e4)
        log1p(adata_combined)
        highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
        # sc.pl.highly_variable_genes(Normal_combined)
        adata_combined = adata_combined[:, adata_combined.var.highly_variable]#highly_variable
        pca(adata_combined)
        neighbors(adata_combined)
        umap(adata_combined)        

        #raise ImportError("")Please install scanpy: `pip install scanpy`.

    return(adata_combined)

def deconvolver(adata_ev, adata_cell, species, OBSsample='batch', OBScelltype='celltype', OBSev='sEV', OBSMpca='X_pca', cellN=10, Xraw = True, normalW=True):

    adata_combined = preprocess_source(adata_ev, adata_cell, OBScelltype=OBScelltype, OBSev=OBSev, Xraw = Xraw)
    gsea_pval_dat = source_biogenesis(adata_cell, species, OBScelltype=OBScelltype, Xraw = Xraw, normalW=normalW)
    near_neighbor_dat = near_neighbor(adata_combined, OBSsample=OBSsample, OBSev=OBSev, OBScelltype=OBScelltype, OBSMpca=OBSMpca, cellN=cellN)
    
    near_neighbor_dat['times'] = ''
    near_neighbor_dat['type'] = ''
    for i in range(near_neighbor_dat.shape[0]):
        ## iteration for all ev
        tmp_times = near_neighbor_dat.iloc[i,2:12].value_counts(sort = True)
        near_neighbor_dat.loc[i, 'times'] = tmp_times[0]
        tmp_keys = near_neighbor_dat.iloc[i,2:12].value_counts(sort = True).keys()
        tmp_times_fil = (tmp_times + gsea_pval_dat.loc[tmp_keys, 'log1p']*4).sort_values(ascending = False)
        near_neighbor_dat.loc[i, 'type'] = tmp_times_fil.keys()[0]

    near_neighbor_dat.index = adata_ev.obs.index
    celltype_e_number = pd.DataFrame(near_neighbor_dat.type.value_counts())

    adata_ev.obsm['source'] = near_neighbor_dat[[0, 1, 'type']]
    adata_ev.obsm['source'].columns = ['sample', 'i', 'type']

    return([celltype_e_number, adata_ev, adata_combined])


def ESAI_celltype(adata_ev, adata_cell, OBSsample='batch', OBScelltype='celltype'):
    ## calculate ESAI in cell type
    ev_type_count_total = pd.DataFrame(adata_ev.obsm['source'].groupby(['sample','type'])['sample'].count())
    cell_type_count = pd.DataFrame(adata_cell.obs.groupby([OBSsample, OBScelltype])[OBSsample].count())

    ev_activity = []
    for i, j in cell_type_count.index:
        if (i,j) in ev_type_count_total.index:
            a = ev_type_count_total.loc[(i, j), 'sample']
            b = cell_type_count.loc[(i, j), OBSsample]
            if b > 0:
                ev_activity.append([i, j, a, b, a / b])
            else:
                print([i, j, a, b, a / b])
        else:
            a = 0
            b = cell_type_count.loc[(i, j), OBSsample]
            ev_activity.append([i, j, a, b, 0])   
              
    ev_activity_dat = pd.DataFrame(ev_activity)
    ev_activity_dat_pivot = pd.pivot_table(ev_activity_dat, index=[0], columns= 1, values=4)

    return(ev_activity_dat_pivot)


from scipy import stats
def density_estimation(m1, m2, methods):
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = stats.gaussian_kde(values, bw_method=methods)                                                               
    Z = np.reshape(kernel(positions).T, X.shape)
    return(X, Y, Z)

def density_adata(adata, methods='silverman'):
    X_umap = adata.obsm['X_umap']
    m1 = X_umap[:,0]
    m2 = X_umap[:,1]
    X, Y, Z = density_estimation(m1, m2, methods=methods)
    return(X, Y, Z)

def plot_SEVumap(adata_com, out_path, plot_cmp='SEV_builtin', save_plot='SEVumap', OBScelltype='celltype', OBSev='sEV', OBSMumap='X_umap',size=10):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import colors
        
        plt.rcParams["figure.figsize"] = (6, 8)
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['pdf.use14corefonts'] = True
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'

        #      = CRC_combined
        celltype_name = adata_com.obs[OBScelltype].values
        #
        if adata_com.obs[OBScelltype].dtype == 'category':
            celltype_name_u = adata_com.obs[OBScelltype].cat.categories#
        else:
            celltype_name_u = celltype_name.unique()
            
        celltype_colorI = dict(map(lambda i,j : (i,j) , celltype_name_u,range(len(celltype_name_u))))
        if plot_cmp == 'SEV_builtin':
            exo_list = ['#8dd3c7', '#ffffb3', '#bebada', '#C8554B', '#66c2a5', '#e78ac3', '#8da0cb', '#fb8072', '#a6d854', '#ffd92f', '#e5c494', '#E7A600']#, '#7c7a28'
            exo_cmap = colors.LinearSegmentedColormap.from_list('exo_cmap', exo_list)
            Tplot_cmp = exo_cmap
        else:
            Tplot_cmp = plot_cmp
        
        sc = plt.scatter(x = adata_com.obsm[OBSMumap][:,0],y=adata_com.obsm[OBSMumap][:,1], c = celltype_name.map(celltype_colorI), cmap = Tplot_cmp, s=size, lw=0, rasterized=True)# alpha=0.6,

        celltype_pos = pd.DataFrame(adata_com.obsm[OBSMumap], columns=["x", "y"]).groupby(adata_com.obs[OBScelltype].values, observed=True).median().sort_index()
        for i, txt in enumerate(celltype_name_u):
            plt.text(celltype_pos.loc[txt,'x'], celltype_pos.loc[txt, 'y'], txt, weight='light', verticalalignment='center', horizontalalignment='center', path_effects=None)

        X, Y, Z = density_adata(adata_com[adata_com.obs[OBScelltype] != OBSev,])
        plt.contour(X, Y, Z, colors  = 'gray', alpha=0.7, zorder=1)

        X, Y, Z = density_adata(adata_com[adata_com.obs[OBScelltype] == OBSev,])
        plt.contour(X, Y, Z, colors  = '#7c7a28', alpha=1, zorder=2, levels=4, linewidths = 1.5)

        plt.tight_layout()
        plt.axis('off')
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.savefig(out_path + str(save_plot) + '.pdf', dpi = 300, transparent=True, format = 'pdf', bbox_inches='tight')
        plt.show()
        print('SEVumap saved in' + out_path + str(save_plot) + '.pdf')
    except ImportError:
        raise ImportError("Please install matplotlib for visualization")

def plot_ESAIumap(adata_com, out_path, obs_ESAI='ESAI_c', plot_cmp='SEV_builtin', save_plot='ESAIumap', OBScelltype='celltype', OBSev='sEV', OBSMumap='X_umap', size=10):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import colors
        
        plt.rcParams["figure.figsize"] = (7.5, 8)
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['pdf.use14corefonts'] = True
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        
        cmap_ESAI = colors.ListedColormap(['#edd1cb', '#eccdc8', '#ebcac5', '#e9c6c2', '#e8c2bf', '#e6bebc', '#e5baba', '#e3b6b7', '#e2b3b5', '#e0afb3', '#deabb0', '#dca7ae', '#daa4ac', '#d8a0aa', '#d69ca8', '#d499a7', '#d295a5', '#cf90a2', '#cc8da1', '#ca899f', '#c7869e', '#c4829c', '#c27f9a', '#bf7c99', '#bc7897', '#b97596', '#b57294', '#b26f93', '#af6c91', '#ab6990', '#a8668e', '#a4638d', '#a1608b', '#9d5d89', '#995a88', '#955786', '#915584', '#8d5282', '#894f80', '#854d7e', '#814a7c', '#7c4879', '#784577', '#724274', '#6e4071', '#693d6f', '#653b6c', '#613969', '#5c3666', '#583463', '#533260', '#4f305c', '#4b2e59', '#462b55', '#422952', '#3d274e', '#39254a', '#352346', '#312042', '#2d1e3e'])
        adata_sev = adata_com[adata_com.obs[OBScelltype] != OBSev,]
        if plot_cmp == 'SEV_builtin':
            Tplot_cmp = cmap_ESAI
        else:
            Tplot_cmp = plot_cmp
            
        sc = plt.scatter(x = adata_sev.obsm[OBSMumap][:,0],y=adata_sev.obsm[OBSMumap][:,1], c = adata_sev.obs[obs_ESAI], \
                    cmap = Tplot_cmp, alpha=0.6, s=size, lw=0, rasterized=True)#sns.cubehelix_palette(as_cmap=True)
        plt.colorbar(sc, aspect=25, fraction=0.04, pad=0.05)#

        X, Y, Z = density_adata(adata_com[adata_com.obs[OBScelltype] != OBSev,])
        plt.contour(X, Y, Z, colors  = 'gray', alpha=0.7, linewidths = 2, zorder=0)

        X, Y, Z = density_adata(adata_com[adata_com.obs[OBScelltype] == OBSev,])
        plt.contour(X, Y, Z, colors  = '#e6a91a', alpha=1, zorder=1, levels=4, linewidths = 2.5)#7c7a28

        plt.tight_layout()
        plt.axis('off')
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.savefig(out_path + str(save_plot) + '.pdf', dpi = 300, transparent=True, format = 'pdf', bbox_inches='tight')
        plt.show()
        print('ESAIumap saved in' + out_path + str(save_plot) + '.pdf')
    except ImportError:
        raise ImportError("Please install matplotlib for visualization")

