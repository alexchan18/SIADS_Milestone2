#feature selection
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
#from clean_data import clean_data
#from retrieve_data import retrieve_data


def get_important_features(data, n_components=0.95):
    """
    Performs PCA on the given data to retain 95% of the variance and returns the most important features.

    Parameters:
    data (pd.DataFrame): The input data.
    n_components (float): The percentage of variance to retain (default is 0.95).

    Returns:
    dict: A dictionary with principal components as keys and lists of important features as values.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA with n_components=0.95
    pca = PCA(n_components=n_components,random_state=42)
    pca.fit(scaled_data)

    # Get the number of components and the explained variance ratio
    n_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f'Number of Components: {n_components}')
    print(f'Explained Variance Ratio: {explained_variance_ratio}')

    # Get the principal component loadings
    loadings = pca.components_

    # Create a DataFrame for the loadings
    loading_df = pd.DataFrame(loadings.T, index=data.columns, columns=[f'PC{i+1}' for i in range(n_components)])
    print(loading_df)

    # Identify top contributing features for each principal component
    top_features = {}
    for i in range(n_components):
        pc = f'PC{i+1}'
        top_features[pc] = loading_df[pc].abs().sort_values(ascending=False).head(5).index.tolist()

    print("Top contributing features for each principal component:")
    for pc, features in top_features.items():
        print(f'{pc}: {features}')
    
    return top_features,loading_df

#return important features
#df = clean_data()
#numeric_cols = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
#data = retrieve_data(df,pos='PG',n_features=numeric_cols)

#important_features, loading_df = get_important_features(data,n_components=0.7)

#important_features
#loading_df
#get the selected features
#selected_features = []
#for list in important_features.values():
#    for feat in list:
#        if feat not in selected_features:
 #           selected_features.append(feat)

#selected_features 
#imp_data = loading_df.T
#desired_columns = ['FG', 'PTS', 'FGA', 'MP', '2P', 'eFG%', 'FG%', '3P%', '2P%', 'FT%']
#imp_data_filtered = imp_data.loc[:, desired_columns]
#imp_data_filtered