import numpy as np
import cupy as cp
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
import random as rn
# import scanpy as sc
# from dca.api import dca
from pyowl import prox_owl
import ipvc
import matplotlib.pyplot as plt

def mydist(w,X, Y):
    dists = -2 * cp.dot(w*X, Y.T) + cp.sum(w*(Y**2),    axis=1) + cp.sum(w*(X**2), axis=1)[:, cp.newaxis]
    # if prob :
    dists[cp.where(dists <0)] = 0
    return dists

def soft_threshold (x,l) :
  a= cp.where(x>=l,-l,0)
  b = cp.where (x<= -l,l,0)
  y = a+b+x
  c = cp.where(y == x,0,y)
  return c

def lwk(dataset,num_clust,beta,lamb,gamma,t) :
    numdata = dataset.shape[0]
    dim = dataset.shape[1]

    # K++
    mu=cp.zeros((num_clust, dim))
    weights = cp.array([1 for i in range(dim)]) 
    mu[0] = dataset[rn.sample(range(numdata),1)[0]]

    for r in range(1,num_clust) :
        data_dist_mat = mydist(weights**beta , dataset,mu[0:r]  )
        data_dist_array = cp.min(data_dist_mat,axis=1)
        mu[r] = dataset[cp.argmax(data_dist_array)]
    
    # K++ -----

    #loop
    cost_last = -cp.inf
    weights = cp.array([1/dim for i in range(dim)]) 
    for l in range(100) :
        # calculate u
        distmatrix = cp.zeros((numdata,num_clust))
        asso_matrix = cp.zeros((numdata,num_clust))
        distmatrix = mydist(weights**beta + (lamb * cp.abs(weights)),dataset,mu)
        min_array = cp.argmin(distmatrix,axis= 1)
        for i in range(numdata) :
            asso_matrix[i,min_array[i]] = 1
        # calculate z
        for clust in range(num_clust) :
            u_clust =asso_matrix[:,clust]
            mu[clust] = cp.zeros((dim))
            mu_data = dataset[cp.where(u_clust != 0)]
            mu[clust] = cp.mean(mu_data,axis=0)
        # claculate W
        a = cp.zeros((dim))
        for clust in range(num_clust) :
            dist_clust = (dataset - mu[clust]) ** 2
            a_add = asso_matrix[:,clust] @ dist_clust
            a = a + a_add
        b = lamb * a
        
        weights = weights - t * beta * weights ** (beta -1) * a
        # print(weights)
        # print(t*(gamma + b))
        # gamma = b-(cp.array(norm.ppf(1-q*np.linspace(1,dim,dim)/dim)))/t
        weights = cp.array(prox_owl(cp.asnumpy(weights),cp.asnumpy(t*(-gamma + b))))
        # print(weights)
        # weights = (1/beta) * soft_threshold(cp.divide(alpha,D),lamb)
        # weights = weights ** (1/(beta-1))
        weights[cp.isinf(weights)] = 0
        cost = cp.dot(weights**beta + (lamb * np.abs(weights)),a) + cp.sum(-gamma * cp.abs(cp.sort(weights)[::-1]))
        if (cost - cost_last)/cost_last <= 1e-5:
            # print(cost - cost_last)
            break
        # print(cost - cost_last)
        cost_last = cost

        computed = cp.argmax(asso_matrix,axis = 1)
        # print(l)
    return cp.asnumpy(computed),cp.asnumpy(weights)

def score_dunn(labels, dataset,weights):
    labels = cp.array(labels)
    clust = cp.unique(labels)
    weights = cp.array(weights)
    distmat = mydist(weights,dataset,dataset)
    inter_clust = cp.zeros(clust.shape[0])
    ext_clust = []
    for i in range(clust.shape[0]):
        inter_clust[i] = cp.max(cp.take(cp.take(distmat,cp.where(clust[i] == labels)[0],axis=0),cp.where(clust[i] == labels)[0],axis=1))
        mu_i = cp.zeros(dataset.shape[1])
        mu_i = cp.mean(dataset[cp.where(clust[i]==labels)[0],:],axis=0)
        for j in range(i):
            mu_j = cp.zeros(dataset.shape[1])
            mu_j = cp.mean(dataset[cp.where(clust[j]==labels)[0],:],axis=0)
            ext_clust.append(cp.linalg.norm(mu_i-mu_j))
    
    return min(ext_clust)/cp.max(inter_clust)

def silhouette(labels, dataset, weights) :
    # labels = np.array(labels)
    dim = dataset.shape[1]
    numdata = dataset.shape[0]
    # weights = np.ones(dim)
    weights = cp.array(weights)
    distmat = mydist(weights,dataset,dataset)
    a = np.zeros(numdata)
    for i in range(numdata) :
        x = distmat[i,np.where(labels == labels[i])[0]]
        a[i] = np.sum(x) / x.shape[0] 
    b = np.zeros(numdata)
    for i in range(numdata) :
        dists = []
        for label in np.unique(labels) :
            if label != labels[i] :
                x = distmat[i,np.where(labels == label)[0]]
                dists.append(np.sum(x) / x.shape[0])
        # print(dists)
        b[i] = min(dists)
    myscore = (b -a) / np.maximum(a,b)
    return sum(myscore)

def mywscore_new(labels,dataset,knn,weights,per = .55) :
    labels = cp.array(labels)
    dim = dataset.shape[1]
    numdata = dataset.shape[0]
    weights = cp.array(weights)
    distmat = mydist(weights,dataset,dataset)
    myscore = 0
    clusters = cp.unique(labels)
    for i in range(dataset.shape[0]):
        cut_score_1 = distmat[i,cp.where(labels != labels[i])[0]]
        cut_score_2 = distmat[i,cp.where(labels == labels[i])[0]]
        k = min(knn,cut_score_1.shape[0],cut_score_2.shape[0])
        myscore +=( cp.sort(cut_score_1)[0:k].sum()*k)
        myscore -=( cp.sort(cut_score_2)[::-1][0:k].sum()*k)
    percentile = cp.amax((cp.unique(labels,return_counts=True)[1])) / labels.shape[0]
    # print(percentile)
    if percentile > per :
        myscore = -np.inf
    return myscore

def mywscore_new2(labels,dataset,knn,weights,per = .55) :
    labels = cp.array(labels)
    dim = dataset.shape[1]
    numdata = dataset.shape[0]
    weights = cp.array(weights)
    distmat = mydist(weights,dataset,dataset)
    myscore = 0
    clusters = cp.unique(labels)
    for i in range(dataset.shape[0]):
        cut_score_1 = distmat[i,cp.where(labels != labels[i])[0]]
        cut_score_2 = distmat[i,cp.where(labels == labels[i])[0]]
        k = min(knn,cut_score_1.shape[0],cut_score_2.shape[0])
        myscore +=( cp.sort(cut_score_1)[0:k].sum()*k) / cp.where(labels == labels[i])[0].shape[0]
        myscore -=( cp.sort(cut_score_2)[::-1][0:k].sum()*k) / cp.where(labels == labels[i])[0].shape[0]
    percentile = cp.amax((cp.unique(labels,return_counts=True)[1])) / labels.shape[0]
    # print(percentile)
    if percentile > per :
        myscore = -np.inf
    return myscore

def myweightipvc(dataset,num_clust,gamma,t,knn,lwk_high) :
    print("last updated 10:09")
    lwk_params= [[6,0.0001],[4,0.0001]]
    if lwk_high is not None :
        lwk_params= [[6,0.0001],[8,0.0001]]
    dim = dataset.shape[1]
    numdata = dataset.shape[0]
#     print(t)
#     score = 0
    total_weight = np.zeros(dim)
    lable = np.zeros((2,numdata))
    for i,[beta,lamda] in enumerate(lwk_params) :
        lable[i],weights = lwk(dataset=dataset ,num_clust = num_clust,beta = beta,lamb=lamda,gamma=gamma,t = t)
        # plt.scatter(np.arange((dim)),weights)
        # plt.show()
        total_weight = total_weight + weights

    final_cluster  = ipvc.ipvc(lables=lable,k = num_clust)
    our_score = mywscore_new(labels = final_cluster.copy(), dataset = dataset , knn = knn, weights =total_weight/2)
    our_score2 = mywscore_new2(labels = final_cluster.copy(), dataset = dataset , knn = knn, weights =total_weight/2)
    dunn_score = score_dunn(labels = final_cluster, dataset = dataset, weights =total_weight )
    silhouette_score = silhouette(labels = final_cluster.copy(), dataset = dataset , weights =total_weight/2 )
    
    return final_cluster, our_score, our_score2, silhouette_score, dunn_score
#     return final_cluster, 0, 0


