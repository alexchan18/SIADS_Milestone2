import pandas as pd
import numpy as np
from clean_data import clean_data
from clust_model import clust_model


def cluster_summary(df):
    """
    Computes means for all fields grouped by the 'cluster' column.

    Args:
        df (pd.DataFrame): Input data frame with a 'cluster' column.

    Returns:
        pd.DataFrame: A new data frame with means, cluster size, and cluster.
    """
    # Group by 'cluster' and calculate means for all columns
    grouped = df.astype(float).groupby('Cluster').mean()

    # Calculate cluster size
    cluster_size = df.groupby('Cluster').size().rename('cluster_size')

    # Merge means and cluster size
    result_df = pd.concat([grouped, cluster_size], axis=1).reset_index()

    return result_df
#cluster size and means
#summary_df = cluster_summary(df_clustered)
#summary_df

