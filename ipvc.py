import numpy as np
import random as rn

def ipvc(lables,k):
    n=lables.shape[1]
    c=lables.shape[0]

    final_cluster=np.array(rn.choices(range(k),k=n))
    old_cluster=final_cluster
    
    for _ in range(100):
        for y in range(n):
            d_main=np.zeros(k)
            for i in range(k):
                d=np.zeros(c)
                iden=np.ones(n)
                ni=(final_cluster==i)
                for j in range(c):
                    d[j]=(lables[j,y]!=lables[j,ni]).sum()
                d_main[i]=np.sum(d)/ni.sum()
            final_cluster[y]=np.argmin(d_main)
        
        if np.all(old_cluster==final_cluster) :
            return final_cluster
        
        old_cluster=final_cluster

    return final_cluster

    

