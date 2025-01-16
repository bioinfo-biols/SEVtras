import numpy as np
import pandas as pd
import anndata
from .sc_readwrite import read, read_10x_h5, read_10x_mtx, write, read_visium
from anndata import AnnData, concat
from anndata import read_h5ad
import scipy
import copy
import sys
import os
import pickle
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count


def get_sample(sample_log):
    names_list = []
    with open(sample_log, 'r') as  f:
        for line in f.readlines():
            line = line.strip()
            names_list.append(line)
    
    return(names_list)


def read_adata(adata_path, dir_origin = True, get_only=False):
    if (adata_path).endswith('.h5'):
        adata = read_10x_h5(adata_path)
    elif (adata_path).endswith('.h5ad'):
        adata = read_h5ad(adata_path)
    else:
        if dir_origin == True:
            adata = read_10x_mtx(adata_path + '/outs/raw_feature_bc_matrix/')
        else:
            adata = read_10x_mtx(adata_path)

    return(adata)

import anndata
def read_project(output_path, names_list):

    names_list_1 = copy.copy(names_list)
    adata_list = []
    for i in (names_list):
        i = i.replace(".", "_")
        if os.path.exists(output_path + '/tmp_out/' + i):
            
            # exec('g{} = read_h5ad("{}/{}/{}/raw_{}.h5ad")'.format(i, str(output_path), 'tmp_out', i, i), d1)
            adata_list.append(read_h5ad(str(output_path) + '/tmp_out/'+ str(i) + '/raw_' + str(i) + '.h5ad'))
            adata_list[-1].var_names_make_unique()
            #d1['g'+i].var_names_make_unique()
            # exec('g{}.var_names_make_unique()'.format(i), globals())
            # exec('adata_list.append(g{})'.format(i))
        else:
            print(i)
            names_list_1.remove(i)
    #d = {}
    #exec('adata_com = adata_list[0].concatenate(d1[g{}], batch_categories = names_list_1)'.format(names_list_1[0], "], d1[g".join(names_list_1[1:])), d)
    #adata_com = d['adata_com']
    #adata_com.write(output_path+'/all.h5ad')
    adata_com = anndata.concat(adata_list, label="batch", keys=names_list_1)#adata_list[0].concatenate(adata_list[1:], batch_categories = names_list_1)
    adata_com.obs_names_make_unique()

    return(adata_com)

def get_genes(output_path, names_list):
    genes_out = []
    for i in names_list:
        path_gene = str(output_path) + '/tmp_out/' + i + '/itera_gene.txt'
        if os.path.exists(path_gene):
            with open (path_gene, 'rb') as fp:
                inter_gene = pickle.load(fp)
                genes_out.append(inter_gene[len(inter_gene)-1])
    
    return(genes_out)

def count_genes(genes_out, out_path):
    cal_gene_out = []
    for i in genes_out:
        cal_gene_out.extend(list(i))

    pd.Series(cal_gene_out).value_counts().head(15).index.to_series().to_csv(str(out_path) + '/ev_genes.csv')
    genes = pd.read_csv(str(out_path) + '/ev_genes.csv')

    return(genes)


def filter_adata(adata):
    adata.var_names_make_unique()
    adata.var['mt'] = adata.var_names.str.startswith(('mt-', 'MT-'))
    adata.var['ribo'] = adata.var_names.str.startswith(('Rps', 'Rpl', 'RPS', 'RPL'))
    try:
        import scanpy as sc
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], inplace=True, percent_top='', log1p =False)
        adata = adata[(adata.obs.pct_counts_mt < 15) & (adata.obs.pct_counts_ribo < 30), :]
        sc.pp.filter_cells(adata, min_genes=6)
        sc.pp.filter_genes(adata, min_cells=5)
        if 'n_genes_by_counts' in adata.obs.columns:
            adata.obs['n_genes'] = adata.obs['n_genes_by_counts']

    except ImportError:
        from .sc_pp import calculate_qc_metrics, filter_cells, filter_genes
        calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], inplace=True, percent_top='', log1p =False)
        adata = adata[(adata.obs.pct_counts_mt < 15) & (adata.obs.pct_counts_ribo < 30), :]
        filter_cells(adata, min_genes=6)
        filter_genes(adata, min_cells=5)
        if 'n_genes_by_counts' in adata.obs.columns:
            adata.obs['n_genes'] = adata.obs['n_genes_by_counts']

    return(adata)

from pathlib import Path
def get_ev_list(species, score_t='15'):
    ## get initial gene set
    ##ev gene symbols
    if species == 'Homo':
        ev_list = [x.strip() for x in open(Path(__file__).parent / 'id_homo_rna_ev')]
    elif species == 'Mus':
        ev_list = [x.strip() for x in open(Path(__file__).parent / 'id_mus_rna_ev')]
        if score_t == '15':
            score_t = '10'

    score_t1 = '1e-' + str(score_t)
    ev_list_count = pd.value_counts(ev_list, sort=False)
    ev_list = ev_list_count[ev_list_count > 0].index
    
    return([ev_list, score_t1])

from scipy.stats import hypergeom
from SEVtras import env
def process_inter(i):
    k = (env.Inter_adata[i, env.Same].X > env.Thershold).sum()
    M = env.Max_M
    n = len(env.Same)
    N = int(env.Inter_adata[i, :].obs['n_genes'])
    return([hypergeom.sf(k, M, n, N), N, k/N, n/M])

def corr_genes(i):
    #i, inter_adata
    if (len(np.unique(env.Inter_adata.X.A[:, i])) > 1):#toarray()
        tmp = scipy.stats.spearmanr(env.Inter_adata.X.A[:, i], -np.log(env.Inter_adata.obs['score'] + 10**-50))#toarray()
        return(tmp.correlation)
    else:
        return(-1)

def multi_enrich(inter_adata, same, iteration_list, thershold, max_M, threads):
    len((iteration_list)), len(same), max(inter_adata.obs['n_genes'])

    out0 = []
    item_list = range(inter_adata.n_obs)
    pool = Pool(threads, env.initializer, (inter_adata, same, thershold, max_M))
    out0 = pool.map(process_inter, item_list)
    pool.close()
    pool.join()

    ##
    out = pd.DataFrame(out0)
    # out.to_csv('./out.csv')
    out['score'] = sm.stats.fdrcorrection(out[0])[1]
    # iteration_list_old = iteration_list
    
    inter_adata.obs['score'] = out['score'].tolist()

    return(inter_adata)


def multi_cor(inter_adata, threads):
    result_tmp = []

    pool = Pool(threads, env.initializer_simple, (inter_adata,))
    item_list = range(inter_adata.X.shape[1])
    
    result_tmp = pool.map(corr_genes, item_list)
    pool.close()
    pool.join()

    return(result_tmp)

def representative_gene(result_tmp, number_g = 30, alpha = 0.10):
    max_number = []
    max_index = []
    t = copy.deepcopy(result_tmp)
    for _ in range(number_g):
        number = max(t)
        index = t.index(number)
        t[index] = -1
        if (number > alpha):
            max_index.append(index)
            max_number.append(number)

    return([max_index, max_number])

def get_iteration(inter_adata, max_index):
    # p_thre_filter = min(max_number)
    #tt_list  = inter_adata.var.index[[(i >= 0) & (i >= p_thre_filter) for i in result_tmp]]
    tt_list = inter_adata.var.index[max_index]
    iteration_list = list(tt_list)# & ev_list_s  set()

    return(iteration_list)

def iteration(inter_adata, iteration_list, thershold, max_M, threads=20, number_g = 30, alpha = 0.10):
    same = [i for i in iteration_list if i in inter_adata.var_names]
    inter_adata = multi_enrich(inter_adata, same, iteration_list, thershold, max_M, threads)
    result_tmp = multi_cor(inter_adata, threads)

    
    max_index, max_number = representative_gene(result_tmp, number_g = 30, alpha = 0.10)
    iteration_list = get_iteration(inter_adata, max_index)

    return(inter_adata, iteration_list)
    
def process(i):
    k = (env.Adata[i, env.Same].X > env.Thershold).sum()
    M = env.Max_M
    n = len(env.Same)
    N = int(env.Adata[i, :].obs['n_genes'])
    return([scipy.stats.hypergeom.sf(k, M, n, N), N, k/N, n/M])


def final_menrich(adata, same, thershold, max_M, threads):

    out0 = []
    item_list = range(adata.n_obs)
    pool = Pool(threads, env.initializer_adata, (adata, same, thershold, max_M))
    out0 = pool.map(process, item_list)
    pool.close()
    pool.join()

    ##
    out = pd.DataFrame(out0)
    out['score'] = sm.stats.fdrcorrection(out[0])[1]
    #iteration_list_old = iteration_list

    adata.obs['score'] = out['score'].tolist()
    
    return(adata)



## cell free droplets simulation
def zinb_genes(i, gene_exp_a, num):
    cells = num #number of cells
    n = 1
    p = 0.8# 1 - droplet rate
    
    exp = gene_exp_a.iloc[i]#exp_i# gene expression mean
    success = 1
    
    out_zinb = np.random.negative_binomial(success, success/(success+exp), size=cells) * \
        np.random.binomial(n, p, size=cells)
    return(pd.Series(data = out_zinb, name = gene_exp_a.index[i]))
    
def run_weigths(gene_exp_a, num=1000):
    
    weight = []
    def log_result(result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        weight.append(result)

    pool = Pool(cpu_count()-4)

    for i in range(len(gene_exp_a)):
        pool.apply_async(zinb_genes, args = (i, gene_exp_a, num), callback = log_result)
        
    pool.close()
    pool.join()
    
#     weight = (gene_exp_a.apply(zinb_genes, axis = 1))
    
    weight = pd.DataFrame(weight)#.T.fillna(0)
    
    return(weight)


def process_random(gene_exp, count):
    num = len(count)
    out = run_weigths(gene_exp, num)
    
    return(round((out / out.sum(0)) * count))






