import pandas as pd
#from clean_data import clean_data

def retrieve_data(data, pos=None, n_features=None):
    """
    Retrieve and filter the dataset based on the specified position and features.

    Parameters
    ----------
    data : DataFrame
        The input dataset.
    pos : str, optional
        The position to filter the dataset. If None, all positions are included.
    n_features : list, optional
        List of feature columns to include in the returned DataFrame. If None, all columns are included.

    Returns
    -------
    DataFrame
        The filtered dataset based on the specified position and features.
    """
    if pos:
        df = data[data['Pos'] == pos]
    else:
        df = data
    
    if n_features:
        # Ensure all n_features are present in the dataframe
        missing_features = [feature for feature in n_features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"The following features are not present in the dataframe: {missing_features}")
        df = df[n_features].copy()
    
    return df

#select only numeric data
#df = clean_data
#numeric_cols = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
#data = retrieve_data(df,pos='PG',n_features=numeric_cols)