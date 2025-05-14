import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
#from clean_data import clean_data
#from evaluate_clustering import evaluate_clustering

def clust_model(data,k=5,pos=None, init_method='k-means++'):
    """
    Clustering model to apply KMeans on the specified position's data and plot the results.
    
    Parameters
    ----------
    data : DataFrame
        The input dataset.
    pos : str
        The position to filter the dataset.
    n_features : list
        List of feature columns to include in the clustering.
    k : int, optional
        Number of clusters. Default is 5.
    init_method : str, optional
        Initialization method for KMeans. Can be 'k-means++', 'random', or 'pca'. Default is 'k-means++'.
    
    Returns
    -------
    DataFrame
        The original data with an additional column for the cluster assignment.
    array
        The scaled data used for clustering.
    """
    #df = data[data['Pos'] == pos]
    df_cluster = data.copy()
    df_cluster =df_cluster.astype(float)
    # Apply standard scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_cluster)
    
    # Handle PCA-based initialization
    if init_method == 'pca':
        pca = PCA(n_components=k)
        pca.fit(data_scaled)
        init_centers = pca.components_
        kmeans = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=42)
    else:
        kmeans = KMeans(n_clusters=k, init=init_method, n_init=25, random_state=42)
    
    # Apply KMeans Clustering
    kmeans.fit(data_scaled)
    df_cluster['Cluster'] = kmeans.predict(data_scaled)

    # Reduce dimensionality using PCA for visualization
    pca_vis = PCA(n_components=2)
    reduced_data = pca_vis.fit_transform(data_scaled)
    
    # Plot the clusters
    plt.figure(figsize=(8, 6))
    for cluster in range(kmeans.n_clusters):
        plt.scatter(reduced_data[df_cluster['Cluster'] == cluster, 0],
                    reduced_data[df_cluster['Cluster'] == cluster, 1],
                    s=50, label=f'Cluster {cluster}')
    
    # Plot the centroids
    centroids = kmeans.cluster_centers_
    centroids_reduced = pca_vis.transform(centroids)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1],
                marker='x', s=100, color='k', label='Centroids')
    
    plt.title(f'K-means clustering on the {pos} position (PCA-reduced data with init {init_method})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    
    return df_cluster, data_scaled


#data = clean_data()
#pos = 'C'
#numeric_cols = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
#numeric_cols = ['G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA','ORB', 'DRB', 'TRB', 'AST', 'BLK', 'TOV', 'PF', 'PTS']
#numeric_cols = ['MP', 'FG%', '3P%', 'AST', 'TOV', 'PTS']
#numeric_cols =['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%']
# Apply clustering
#df_clustered, data_scaled = clust_model(x_data,k=3, init_method='pca')
#df_clustered, data_scaled = clust_model(x_data,k=3, init_method='k-means++')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='random')

#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='pca')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='k-means++')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='random')






























#def clust_model(data,pos,n_features):
#    df = data[data['Pos']==pos]
#    df_cluster=df[n_features].copy()
#    #apply standard scale
#    scaler = StandardScaler()
#    data_scaled = scaler.fit_transform(df_cluster)
#    # Applying KMeans Clustering
#    kmeans = KMeans(n_clusters=5, random_state=42)
#    df_cluster['Cluster'] = kmeans.fit_predict(data_scaled)
#    return df_cluster


#def clust_model(data, pos, n_features,k=4):
#    df = data[data['Pos'] == pos]
#    df_cluster = df[n_features].copy()
#    
#    # Apply standard scaling
#    scaler = StandardScaler()
#    data_scaled = scaler.fit_transform(df_cluster)
#    
#    # Applying KMeans Clustering
#    kmeans = KMeans(n_clusters=k, random_state=42)
#    kmeans.fit(data_scaled)
#    df_cluster['Cluster'] = kmeans.predict(data_scaled)
#    
#    # Reduce dimensionality using PCA
#    pca = PCA(n_components=2)
#    reduced_data = pca.fit_transform(data_scaled)
#    
#    # Plot the clusters
#    plt.figure(figsize=(8, 6))
#    for cluster in range(kmeans.n_clusters):
#        plt.scatter(reduced_data[df_cluster['Cluster'] == cluster, 0],
#                    reduced_data[df_cluster['Cluster'] == cluster, 1],
#                    s=50, label=f'Cluster {cluster}')
#    
#    # Plot the centroids
#    centroids = kmeans.cluster_centers_
#    centroids_reduced = pca.transform(centroids)
#    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1],
#                marker='x', s=100, color='k', label='Centroids')
#    
#    plt.title(f'K-means clustering on the {pos} position (PCA-reduced data)')
#    plt.xlabel('Principal Component 1')
#    plt.ylabel('Principal Component 2')
#    plt.legend()
#    plt.show()
#    
#    return df_cluster

# Assuming you have a function 'clean_data()' that returns your data



