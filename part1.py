import myplots as myplt
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
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data,labels,num_of_clusters,random_init=42):
    kmm=KMeans(n_clusters=num_of_clusters,random_state=random_init)
    ss=StandardScaler()
    train=ss.fit_transform(data)
    kmm.fit(train,labels)
    preds=kmm.predict(train)
    return preds


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    random_state = 42

    noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05, random_state=random_state)

    varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    aniso_data, aniso_labels = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = (np.dot(aniso_data, transformation), aniso_labels)

    blobs = datasets.make_blobs(n_samples=100, random_state=random_state)

    dct = answers["1A: datasets"] = {"nc": [noisy_circles[0], noisy_circles[1]],
    "nm": [noisy_moons[0], noisy_moons[1]],
    "bvv": [varied[0], varied[1]],
    "add": [aniso[0], aniso[1]],
    "b": [blobs[0], blobs[1]]}

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """ 

    dct = answers["1B: fit_kmeans"] = fit_kmeans
    fit_result=dct

    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    Kmeans_dict_plotting = {}
    for dataset_name, dataset_info in answers['1A: datasets'].items():
        acc = []
        dataset_cluster = {}
        for num_clusters in [2, 3, 5, 10]:
            preds = dct(dataset_info[0], dataset_info[1], num_clusters, 42)
            dataset_cluster[num_clusters] = preds
        acc.append(dataset_info)
        acc.append(dataset_cluster)
        Kmeans_dict_plotting[dataset_name] = acc

    myplt.plot_part1C(Kmeans_dict_plotting, 'plotpart1-C.jpg')

    dct = answers["1C: cluster successes"] = {"bvv": [3], "add": [3],"b":[3]}
    dct = answers["1C: cluster failures"] = {"nc","nm"}


    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    Kmeans_dict_plotting = {}
    for dataset_name, dataset_info in answers['1A: datasets'].items():
        acc = []
        dataset_cluster = {}
        for num_clusters in [2, 3]:
           preds = fit_result(dataset_info[0], dataset_info[1], num_clusters, 42)
           dataset_cluster[num_clusters] = preds
        acc.append(dataset_info)
        acc.append(dataset_cluster)
        Kmeans_dict_plotting[dataset_name] = acc

    myplt.plot_part1C(Kmeans_dict_plotting, 'plotpart1-D.jpg')

    dct = answers["1D: datasets sensitive to initialization"] = ["nc","nm"]
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
