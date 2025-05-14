import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from clean_data import clean_data
from evaluate_clustering import evaluate_clustering
from cluster_summary import cluster_summary
from clust_model import clust_model
from retrieve_data import retrieve_data
from get_important_features import get_important_features
from find_optimal_k import plot_elbow_method
from find_optimal_k import silhouette_method



# Load and clean the data
df = clean_data()

#return important features
#df = clean_data()
numeric_cols = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
data = retrieve_data(df,pos='PG',n_features=numeric_cols)

important_features, loading_df = get_important_features(data,n_components=0.7)

#important_features
#loading_df
#get the selected features
selected_features = []
for list in important_features.values():
    for feat in list:
        if feat not in selected_features:
            selected_features.append(feat)
#check PCA summary
#selected_features 
imp_data = loading_df.T
desired_columns = ['FG', 'PTS', 'FGA', 'MP', '2P', 'eFG%', 'FG%', '3P%', '2P%', 'FT%']
imp_data_filtered = imp_data.loc[:, desired_columns]
imp_data_filtered

# Define the position and features
#Find optimal K 
pos = 'PG'
selected_feat = ['FG', 'PTS', 'FGA', 'MP', '2P', 'eFG%', 'FG%', '3P%', '2P%', 'FT%']
x_data = retrieve_data(df,pos=pos,n_features=selected_feat)

#plot charts to find optimal k
plot_elbow_method(x_data , max_k=10)

silhouette_method(x_data)




# Perform clustering
x_data = retrieve_data(df,pos='PG',n_features=selected_feat)
#df_clustered, data_scaled = clust_model(data, pos=pos, n_features=numeric_cols, k=6, init_method='k-means++')
df_clustered, data_scaled = clust_model(x_data,k=3,pos='PG',init_method='k-means++')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='random')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='pca')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='k-means++')
#df_clustered, data_scaled = clust_model(x_data,k=4, init_method='random')


# Extract true labels and cluster labels
true_labels = df[df['Pos'] == pos]['Pos'].factorize()[0]
cluster_labels = df_clustered['Cluster']

# Evaluate clustering
evaluation_results = evaluate_clustering(true_labels, cluster_labels, data_scaled)

# Print evaluation results
print(f"\nEvaluation metrics for k={6}:")
for metric, score in evaluation_results.items():
    print(f"{metric}: {score:.4f}")

# Print the first few rows of the clustered DataFrame
print(df_clustered.head())

# Get and print cluster summary
summary_df = cluster_summary(df_clustered)
print(summary_df)