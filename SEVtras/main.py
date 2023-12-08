from .utils import *
import os
import sys

def SEVtras_command():
    ''' Example of taking inputs for SEVtras'''
    args = sys.argv[1:]
    if len(args) < 1:
        print("usage: sEV_recognizer, ESAI_calculator, cellfree_simulator, sEV_enrichment")


def cellfree_simulator(out_path, gene_exp_ev, gene_exp_cell, expect_UMI = [40, 70, 100, 130], sEV_fraction = [0.005, 0.01, 0.05, 0.10], sEV=500):
    all_num = [sEV/f for f in sEV_fraction]
    for i in expect_UMI:
        for j in all_num:
            ev_count = np.random.poisson(lam=i, size=sEV)
            all_count = np.random.poisson(lam=i, size=j)

            ev_data = process_random(gene_exp_ev, ev_count)
            ev_data.columns = ['ev'+str(i) for i in ev_data.columns.values]
            cell_data = process_random(gene_exp_cell, all_count)
            ev_cell_data = pd.concat([cell_data, ev_data], axis=1).fillna(0)

            new_index = [ i.split(' /// ')[-1].split('-')[0] for i in ev_cell_data.index.values]
            ev_cell_data.index = new_index
            ev_cell_data.to_csv(out_path + '/simulation_UMI' + str(i) + '_fraction' + str(int(sEV/j)) + '.csv')
    
    return('Simulation done!')

def sEV_aggregator(out_path, name_list, max_M, score_t, threads, search_UMI, flag=0):
    if flag == 0:
        adata_com = read_project(out_path, name_list)
        # adata_com.write(output_path+'/all.h5ad')
        genes_out = get_genes(out_path, name_list)
        genes = count_genes(genes_out, out_path)
        iteration_list = set(genes['0'])#ev_list_s
        inter_gene = []
        inter_out_p = pd.DataFrame(list(range(0, adata_com.obs.shape[0])))
        pth = 0.005
        same = [i for i in (iteration_list) if i in adata_com.var_names]
        thershold = 0
        max_M = max(adata_com.obs['n_genes'])
        adata_com = final_menrich(adata_com, same, thershold, max_M, threads)

        adata_com.obs['sEV'] = ['sEV' if i else 'False' for i in (adata_com.obs['total_counts'] < search_UMI) & (adata_com.obs['score'] < score_t)]

        adata_com.obs['score1'] = -np.log10(adata_com.obs['score'] + adata_com.obs.loc[(adata_com.obs['score']>0).values, 'score'].min()/10)
        ## write files in this project
        adata_ev = copy.copy(adata_com[(adata_com.obs['total_counts'] < search_UMI) & (adata_com.obs['score'] < score_t), :])
        adata_ev = adata_ev.copy()
        adata_com.obs['score'] = adata_com.obs['score1']
        adata_ev.obs['score'] = adata_ev.obs['score1']

        adata_com.write(str(out_path) + '/raw_' + 'SEVtras' + '.h5ad')
        adata_ev.write(str(out_path) + '/sEV_SEVtras.h5ad')
    
    return('Aggregation done!')


def sEV_recognizer(sample_file, out_path, input_path=None, species='Homo', predefine_threads=-2, get_only=False, score_t = None, search_UMI=500, alpha=0.10, dir_origin=True):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if predefine_threads == -2:
        threads = cpu_count()-2
    else:
        threads = predefine_threads
    
    if out_path.endswith('/'):
        out_path = out_path[:-1]
    
    if score_t is not None:
        ev_list, score_t = get_ev_list(species, score_t)
    else:
        ev_list, score_t = get_ev_list(species)
    score_t = float(score_t)
    ev_list_s = list(set(ev_list))
    
    sample_log = get_sample(sample_file)
    len_sample = len(sample_log)
    name_list = []

    for sample in sample_log:
        ##read adata
        if input_path!= None:
            each_adata = str(input_path) + '/' + str(sample)
        else:
            each_adata = sample

        adata = read_adata(each_adata, get_only=get_only, dir_origin=dir_origin)
        adata = filter_adata(adata)
        sample = sample.replace(".", "_")
        name_list.append(sample)

        iteration_list = ev_list_s
        inter_gene = []
        pth = 0.005
        flag = 1
        inter_out_p = pd.DataFrame(list(range(0, adata.obs.shape[0])))
        inter_adata = copy.copy(adata[(adata.obs['total_counts'] < 200) & (adata.obs['total_counts'] > 20),:])
        inter_adata = inter_adata.copy()
        try:
            import scanpy as sc
            sc.pp.normalize_total(inter_adata, target_sum=1e2)
        except ImportError:
            from .sc_pp import normalize_total
            normalize_total(inter_adata, target_sum=1e2)
            
        max_M = max(adata.obs['n_genes'])
        # inter_adata.X = (inter_adata.X / inter_adata.X.sum(1).reshape(inter_adata.shape[0], 1))*100
        ##iterations 
        for itera in range(10):
            if(len(iteration_list) == 0):
                print('No genes enriched, please check input sample:' + str(sample))
                flag = 0
                break

            thershold = 0
            iteration_list_old = iteration_list
            inter_adata, iteration_list = iteration(inter_adata, iteration_list, thershold, max_M, threads=threads, number_g = 30, alpha = alpha)
            print(itera, len(iteration_list))
                
            if(len(set(iteration_list_old) & set(iteration_list)) == max(len(iteration_list_old), len(iteration_list))):
                inter_gene.append(iteration_list)
                break
            else:
                inter_gene.append(iteration_list_old)
                inter_out_p = pd.concat((inter_out_p, inter_adata.obs['score']), axis=1, ignore_index=True)


        if flag == 1:
            genes = iteration_list[0:15]
            same = [i for i in (genes) if i in adata.var_names]
            thershold = 0
            adata = final_menrich(adata, same, thershold, max_M, threads)

        elif (len(iteration_list_old) > 0) & (len(iteration_list_old) < len(ev_list_s)):
            genes = iteration_list_old[0:15]
            same = [i for i in (genes) if i in adata.var_names]
            thershold = 0
            adata = final_menrich(adata, same, thershold, max_M, threads)

        else:
            print('sEV is hard to detect in ' + str(sample))
            # continue
        
        ## write files
        if len_sample > 1:
            if os.path.isdir(str(out_path) + '/tmp_out'):
                pass
            else:
                os.mkdir(str(out_path) + '/tmp_out')

            if os.path.isdir(str(out_path) + '/tmp_out/' + sample):
                pass
            else:
                os.mkdir(str(out_path) + '/tmp_out/' + sample)

            adata.write(str(out_path) + '/tmp_out/' + sample + '/raw_' + sample + '.h5ad')
            
            print(str(sample) + ' finished')
            if len(inter_gene) > 0:
                with open(str(out_path) +'/tmp_out/' + sample + '/itera_gene.txt', 'wb') as fp:
                    pickle.dump(inter_gene, fp)

        else:
            adata.obs['sEV'] = ['sEV' if i else 'False' for i in (adata.obs['total_counts'] < search_UMI) & (adata.obs['score'] < score_t)]
            adata.obs['score'] = -np.log10(adata.obs['score'] + adata.obs.loc[(adata.obs['score']>0).values, 'score'].min()/10)
            adata.write(str(out_path) + '/raw_' + str(sample) + '.h5ad')
            with open(str(out_path)+ '/' + str(sample) + "_ev.txt", "w") as f:
                for s in genes:
                    f.write(str(s) +"\n")

    ## Aggregation
    if len_sample > 1:
        sEV_aggregator(out_path, name_list, max_M, score_t, threads, search_UMI)

    return('Recognization done!')

def ESAI_calculator(adata_ev_path, adata_cell_path, out_path, OBSsample='batch', OBScelltype='celltype', OBSev='sEV', OBSMpca='X_pca', cellN=10, Xraw = True, normalW=True, plot_cmp='SEV_builtin', save_plot_prefix='', OBSMumap='X_umap',size=10):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    adata_ev = read_adata(adata_ev_path, get_only=False)
    adata_cell = read_adata(adata_cell_path, get_only=False)
    from .functional import deconvolver, ESAI_celltype, plot_SEVumap, plot_ESAIumap
    celltype_e_number, adata_evS, adata_com = deconvolver(adata_ev, adata_cell, OBSsample, OBScelltype, OBSev, OBSMpca, cellN, Xraw, normalW)
    ##ESAI for sample
    sample_ESAI = (adata_com[adata_com.obs[OBScelltype]==OBSev,].obs[OBSsample].value_counts() / adata_com[adata_com.obs[OBScelltype]!=OBSev,].obs[OBSsample].value_counts()).fillna(0)

    ev_activity_dat_pivot = ESAI_celltype(adata_evS, adata_cell, OBSsample, OBScelltype)

    out_path = str(out_path)
    if out_path.endswith('/'):
        pass
    else:
        out_path = out_path + '/'

    ##save
    adata_evS.write(out_path + save_plot_prefix + 'SEVtras_sEVs' + '.h5ad')
    
    sample_ESAI.to_csv(out_path + save_plot_prefix + 'ESAI_sample.csv')
    ev_activity_dat_pivot.to_csv(out_path + save_plot_prefix + 'ESAI_celltype.csv')
    

    plot_SEVumap(adata_com, out_path, plot_cmp, save_plot=save_plot_prefix+'SEVumap', OBScelltype=OBScelltype, OBSev=OBSev, OBSMumap=OBSMumap,size=size)

    # tmp_dat = exo_activity_dat.groupby('1').sum()#
    adata_com.obs['ESAI_c'] = adata_com[adata_com.obs[OBScelltype] != OBSev,].obs[OBScelltype].map(ev_activity_dat_pivot.median())#

    out = []
    for i in adata_com.obs[[OBSsample,OBScelltype]].values:
        batchI, celltypeI = i
        if celltypeI != OBSev:
            out.append(ev_activity_dat_pivot.loc[batchI, celltypeI])
        else:
            out.append(-1)
    adata_com.obs['ESAI_cS'] = out

    plot_ESAIumap(adata_com, out_path, obs_ESAI='ESAI_c', plot_cmp=plot_cmp, save_plot=save_plot_prefix+'ESAIumap', OBScelltype=OBScelltype, OBSev=OBSev, OBSMumap=OBSMumap,size=size)
    plot_ESAIumap(adata_com, out_path, obs_ESAI='ESAI_cS', plot_cmp=plot_cmp, save_plot=save_plot_prefix+'ESAIumap_sample', OBScelltype=OBScelltype, OBSev=OBSev, OBSMumap=OBSMumap,size=size)

    ##save
    adata_com.write(out_path + save_plot_prefix + 'SEVtras_combined' + '.h5ad')

    return('ESAI done!')

def sEV_imputation(adata_sEV):
    from .sc_pp import magic
    adata_out = magic(adata_sEV)
    print('Imputation done')
    return(adata_out)

def sEV_enrichment(adata_sEV, nBP=15):
    matrix = adata_sEV.X.A
    names = adata_sEV.var_names
    n_top = 15
    norm_dict = (matrix/(1+matrix.sum(1).reshape(len(matrix), 1)))*100
        
    mean_percent = np.median(norm_dict, axis=0)#.mean(axis=0)
    top_idx = np.argsort(mean_percent)[::-1][:n_top]
    # counts_top_genes = norm_dict[:, top_idx]
    columns = (names[top_idx])
    from .sc_utils import enrich
    go_cc = enrich(columns.to_list(), gprofiler_kwargs = {'sources':['GO:CC']})[0:nBP]

    return(go_cc)
