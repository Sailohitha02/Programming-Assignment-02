from pprint import pprint

import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

model_sse_inertial={}
model_sse_manual={}
def fit_kmeans(X, k):
    sse_inertia = []
    sse_man = []

    for clusters_num in range(1, k + 1):
        kmeans = KMeans(n_clusters=clusters_num)
        preds = kmeans.fit_predict(X)

        sse = np.zeros(clusters_num)  

        for cluster_idx in range(clusters_num):
            cluster_points = X[preds == cluster_idx]  
            sse[cluster_idx] = np.sum((cluster_points - kmeans.cluster_centers_[cluster_idx]) ** 2)

        sse_inertia.append(kmeans.inertia_)  
        sse_man.append(np.sum(sse))  

        model_sse_inertial[clusters_num] = kmeans.inertia_  
        model_sse_manual[clusters_num] = np.sum(sse)  

    return sse_inertia, sse_man



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    from sklearn.datasets import make_blobs

    center_box = (-20, 20)
    n_samples = 20
    centers = 5
    random_state = 12

    data, labels = make_blobs(n_samples=n_samples, centers=centers, center_box=center_box, random_state=random_state)

    data, labels

    cord_1=data[0:,0:1]
    cord_2=data[0:,1:]

    dct = answers["2A: blob"] = [cord_1,cord_2,labels]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    X=np.concatenate([answers["2A: blob"][0],answers['2A: blob'][1]],axis=1)
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    
    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    
    sse_c=fit_kmeans(X,k=8)[1]
    sse_vs_k=[[x,y] for x,y in zip(range(1,9),sse_c)]
    plt.plot(np.array(sse_vs_k)[:,1])
    plt.title('K-Value vs SSE')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE: Sum of Squared Errors')
    plt.xticks(range(1, 9))
    plt.grid(True)
    plt.savefig("plotpart2-C.png")

    dct = answers["2C: SSE plot"] =sse_vs_k


    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    sse_d=fit_kmeans(X,k=8)[0]
    sse_vs_k=[[x,y] for x,y in zip(range(1,9),sse_d)]
 
    dct = answers["2D: inertia plot"] = sse_vs_k
    

    dct = answers["2D: do ks agree?"] = "no"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
