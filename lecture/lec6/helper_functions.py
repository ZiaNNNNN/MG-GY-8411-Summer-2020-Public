import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster 
from sklearn import metrics

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

def find_nearest(point, centers):
    distances = np.sum((centers - point)**2, axis = 1)
    return np.min(distances)

def compute_metrics(model, data):
    data = data.copy()
    data["label"] = model.labels_

    data["distance"] = data.apply(lambda row : find_nearest(row.values[:-1], model.cluster_centers_), axis = 1)

    inertia = data["distance"].sum()

    distortion = data.groupby(data["label"]).agg({"distance" : np.mean}).sum().values[0]

    silhouette = metrics.silhouette_score(data, model.labels_)
    
    return inertia, distortion, silhouette

def generate_chart_coordinates(model, data):
    try:
        centers = pd.DataFrame(model.cluster_centers_)
        centers.columns = data.columns
    except:
        data_copy = data.copy()
        data_copy["index"] = model.labels_
        centers = data_copy.groupby("index").mean()
        
    center = data.mean()        
        
    centers = centers - center
    centers = centers.reset_index()
    centers = centers.rename(columns={"index" : "Labels"})

    centers_unpivot = pd.melt(centers, id_vars="Labels", var_name="Category", value_name="Count")
    
    sns.barplot(data=centers_unpivot, x = "Category", y = "Count", hue = "Labels", ci = 0);
    

def generate_dendrogram(data):
    #This function gets pairwise distances between observations in n-dimensional space.
    dists = pdist(data)

    #This function performs hierarchical/agglomerative clustering on the condensed distance matrix y.
    links = linkage(dists, method = "ward")

    p = 46
    #Now we want to plot the dendrogram
    fig = plt.figure(figsize = (12, 6))
    den = dendrogram(links, truncate_mode = 'lastp', p = p)
    plt.xlabel('Employees')
    plt.ylabel('Distance')
    plt.suptitle('Agglomerative Clustering')
    
    return links
    