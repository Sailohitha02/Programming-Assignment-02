import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
import myplots as myplt

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data,labels,k,linkage='ward'):
    ss=StandardScaler()
    ac = AgglomerativeClustering(n_clusters=k,linkage=linkage)
    data = ss.fit_transform(data)
    ac.fit(data,labels)
    preds = ac.labels_
    return preds

def fit_hierarchical_cluster_2(data,labels,link_type,k):
    ss=StandardScaler()
    ac = AgglomerativeClustering(n_clusters=k,linkage=link_type)
    data = ss.fit_transform(data)
    ac.fit(data,labels)
    preds = ac.labels_
    return preds

def fit_modified(data,labels,link_type):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    Z = linkage(data,link_type)
    print(Z)
    calc_distance = []
    for i in range(len(Z)-1):
        calc_distance.append(Z[i+1][2]-Z[i][2])
    ac = AgglomerativeClustering(n_clusters=None,linkage=link_type,distance_threshold=Z[np.argmax(calc_distance),2])  
    ac.fit(data,labels)
    labels = ac.labels_
    return labels

def compute():
    answers = {}
    random_state = 42

    # Generating the datasets
    noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05, random_state=random_state)
    varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    aniso_data, aniso_labels = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = (np.dot(aniso_data, transformation), aniso_labels)
    blobs = datasets.make_blobs(n_samples=100, random_state=random_state)
    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {"nc": [noisy_circles[0], noisy_circles[1]],
    "nm": [noisy_moons[0], noisy_moons[1]],
    "bvv": [varied[0], varied[1]],
    "add": [aniso[0], aniso[1]],
    "b": [blobs[0], blobs[1]],}
   
    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    ac_value = {}
    for dataset_name, dataset_info in answers["4A: datasets"].items():
        dataset_cluster = {}
        for cluster_type in ['single', 'complete', 'ward', 'average']:
            preds = fit_hierarchical_cluster_2(dataset_info[0], dataset_info[1], cluster_type, 2)
            dataset_cluster[cluster_type] = preds
        ac_value[dataset_name] = [dataset_info, dataset_cluster]

    myplt.plot_part1C(ac_value, 'plotpart4-B.jpg')

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]


    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
   
    dct = answers["4C: modified function"] = fit_modified

    ac_value = {}
    for dataset_name, dataset_info in answers["4A: datasets"].items():
        dataset_cluster = {}
        for cluster_type in ['single', 'complete', 'ward', 'average']:
            preds = fit_modified(dataset_info[0], dataset_info[1], cluster_type)
            dataset_cluster[cluster_type] = preds
        ac_value[dataset_name] = [dataset_info, dataset_cluster]

    myplt.plot_part1C(ac_value, 'plotpart4-C.jpg')
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
