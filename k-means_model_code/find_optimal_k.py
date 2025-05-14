import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
#from clean_data import clean_data
#from retrieve_data import retrieve_data

def plot_elbow_method(data, max_k=10):
    """
    Plots the Elbow Method graph to help identify the optimal number of clusters for k-means.

    Parameters:
    data (array-like or sparse matrix): The input data to cluster.
    max_k (int): The maximum number of clusters to test. Default is 10.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    inertia = []
    K = range(1, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k,init='k-means++', n_init=25, max_iter=100,random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()






#df = clean_data()
#pos = 'PG'
#numeric_cols = ['FG', 'PTS', 'FGA', 'MP', '2P', 'eFG%', 'FG%', '3P%', '2P%', 'FT%']
#x_data = retrieve_data(df,pos=pos,n_features=numeric_cols)

#plot_elbow_method(x_data , max_k=10)
#silhouette_method(x_data)

#using the silhouette method to find optimal k
def silhouette_method(data):

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    silhouette_scores = []
    K = range(2, 11)  # Silhouette score is undefined for k=1
    for k in K:
        kmeans = KMeans(n_clusters=k,init='k-means++', n_init=25, max_iter=100,random_state=42)
        kmeans.fit(data_scaled)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(data_scaled, labels))

    # Plot the Silhouette graph
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()

#silhouette_method(x_data)