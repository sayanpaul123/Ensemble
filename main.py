import numpy as np
import cupy as cp
import os
import pandas as pd
# from scanpy.plotting._tools import pca_loadings
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as ami
import random as rn
# import scanpy as sc
# from dca.api import dca
from pyowl import prox_owl
from lwk_slope import lwk, myweightipvc, mywscore_new2, mywscore_new, silhouette
import ipvc
import time

def  lwkSlope(dataset,reduced = False, reduce = False,reduce_name = "", lables = None,num_clust = None,min_counts=3,threads = 2,norm_pp = True, q = 0.9, t = 2, iteration = 30, rep_iter = 5, knn = 80, log1p_pp = True, auth_norm = True, auth_log = 100, tsne_plot = False,pca_plot = False,dca_red = True,auth_dim = True,gamma_opt = None,lwk_high = None):
    print("last updated 10:09")
    if not reduced :
        print("Applying gene filtration")
        dataset = sc.AnnData(pd.DataFrame(dataset))
        sc.pp.filter_genes(dataset, min_counts=min_counts)
        if dca_red == True :
            dca(dataset,threads=threads)
        if lables is not None:
            dataset.obs = pd.DataFrame({ 'Ground Truth':lables})
            # cell_type = list(pd.DataFrame(lables))[0]
            adata = dataset
            if tsne_plot == True:
                sc.tl.tsne(adata)
                sc.pl.tsne(adata,color = 'Ground Truth')
            if pca_plot == True :
                sc.pp.pca(adata)
                sc.pl.pca_scatter(adata, color = 'Ground Truth')
            # del adata
        ## Author's special treatment
        if norm_pp == True:
            print("normalizing per cell")
            sc.pp.normalize_per_cell(dataset)
            # sc.pp.normalize_total(dataset,target_sum = 1)
        if log1p_pp == True:
            print("Applying log transformation")
            sc.pp.log1p(dataset)
        dataset = cp.array(dataset.X)
        print("Dimension of the dataset:",dataset.shape)
        if auth_dim == True :
            print("reducing dimension")
            dims = []
            for  i in range(dataset.shape[1]):
                if not ( np.count_nonzero( dataset[:,i] )) <= 0.06*dataset.shape[0] :
                    dims.append(i)
            dataset = dataset[:,dims]

        # dataset = normalize(dataset, axis = 1,norm = 'l1')
        # 
        if auth_norm == True :
            print("Applying normalization")
            dataset = (dataset.T/dataset.sum(axis=1)).T
        if auth_log > 0:
            print("Applying log transformation with multiplyer")
            dataset = np.log(dataset*auth_log + 1 )
    if reduce :
        print("Saving transformed data")
        name = reduce_name
        if norm_pp :
            name = name + "_norm"
        if log1p_pp :
            name = name + "_log"
        np.savetxt(name+".txt", cp.asnumpy(dataset))
        return

    dataset = cp.array(dataset)
    lables = lables.apply(list(lables).index)
    lables = lables.to_numpy()
    # lables = cp.array(lables,dtype=int)
    numdata = dataset.shape[0]
    dim = dataset.shape[1]

    ## Initialisation of gamma
    gamma = (0.1**2) * cp.linspace(1,dim,dim)/dim
    if gamma_opt is not None:
        gamma = (0.5)*(cp.linspace(dim,1,dim)/dim)**0.5
    result1 = np.zeros((rep_iter,4))
    result2 = np.zeros((rep_iter,4))
    result3 = np.zeros((rep_iter,4))

    number_of_candidates = iteration

    for rep in range(rep_iter) :
        
        
        # max_labels = myipvc(dataset) # not resampling
        individual_labels = np.zeros((number_of_candidates,numdata))
        our_max_score = -np.inf
        our_max_score2 = -np.inf
        dunn_max_score = -np.inf
        silhouette_max_score = -np.inf
        our_best_lables = np.zeros(numdata)
        our_best_lables2 = np.zeros(numdata)
        dunn_best_lables = np.zeros(numdata)
        silhouette_best_lables = np.zeros(numdata)
        start_time = time.time()
        for u in range(number_of_candidates) :
            
            individual_labels[u],our_score,our_score2, silhouette_score, dunn_score = myweightipvc(dataset,num_clust = num_clust,gamma=gamma,t=t,knn=knn,lwk_high = lwk_high)
            
            print(u,ari(individual_labels[u],cp.asnumpy(lables)))
            if our_score > our_max_score : 
                our_max_score = our_score
                our_best_lables = individual_labels[u]
            if dunn_score > dunn_max_score : 
                dunn_max_score = dunn_score
                dunn_best_lables = individual_labels[u]
            if our_score2 > our_max_score2 : 
                our_max_score2 = our_score2
                our_best_lables2 = individual_labels[u]

            if silhouette_score > silhouette_max_score : 
                silhouette_max_score = silhouette_score
                silhouette_best_lables = individual_labels[u]
            
        print("Total time:",rep,time.time()-start_time)
        # best_lables = best_best(individual_labels,dataset,49)

        # adata.obs = pd.DataFrame({'rep' : np.char.add('cluster ',our_best_lables.astype(int).astype(str))})
        # color = list(pd.DataFrame(our_best_lables))[0]
        # # print(adata.obs)
        # # print(adata)
        # if tsne_plot == True:
        #     # sc.tl.tsne(adata)
        #     sc.pl.tsne(adata,color = 'rep')
        # if pca_plot == True :
        #     # sc.pp.pca(adata)
        #     sc.pl.pca_scatter(adata, color = 'rep')
        result1[rep] = np.array([ari(our_best_lables,lables),ari(our_best_lables2,lables),ari(dunn_best_lables,lables),ari(silhouette_best_lables,lables)])
        result2[rep] = np.array([ami(our_best_lables,lables),ami(our_best_lables2,lables),ami(dunn_best_lables,lables),ami(silhouette_best_lables,lables)])
        result3[rep] = np.array([nmi(our_best_lables,lables),nmi(our_best_lables2,lables),nmi(dunn_best_lables,lables),nmi(silhouette_best_lables,lables)])
        print("max ari should be")
        print(result1[rep])
        # print(result2[t])
        # print(result3[t])
        print("Step no:",rep)
    # print("\n\nmax")
    # print(max(result1),max(result2),max(result3))


    print("results are in order our score, our score 2, dunn index, silhouette index")
    print("avg ARI is:",np.mean(result1,axis=0))
    print("max ARI is:", np.max(result1,axis=0))
    print("min ARI is:", np.min(result1,axis = 0))
    print("sd ARI is:", np.std(result1,axis=0))

    print("avg AMI is:",np.mean(result2,axis=0))
    print("max AMI is:", np.max(result2,axis=0))
    print("min AMI is:", np.min(result2,axis = 0))
    print("sd AMI is:", np.std(result2,axis=0))

    print("avg NMI is:",np.mean(result3,axis=0))
    print("max NMI is:", np.max(result3,axis=0))
    print("min NMI is:", np.min(result3,axis = 0))
    print("sd NMI is:", np.std(result3,axis=0))
