import pandas as pd

#input_file = 'NBA_2024_per_game_stats.csv'

def clean_data(input_file='NBA_2024_per_game_stats.csv'):
    df = pd.read_csv(input_file)
    print("Length before cleaning:", len(df))
    
    # Remove rows with repeated headers
    df = df[df['Rk'] != 'Rk']
    df.reset_index(drop=True, inplace=True)
    # Replace NaN values with 0 in specified percentage columns
    #percentage_columns = ['FG%', '3P%', '2P%', 'eFG%', 'FT%']
    # df[percentage_columns] = df[percentage_columns].fillna(0)
    df.fillna(0, inplace=True)
    # Convert data type of percentage columns to float
    #for col in percentage_columns:
    #    df[col] = df[col].astype(float)
    # Convert data type of other numeric columns from object to int
    #cat_columns = ['Rk', 'Player', 'Pos', 'Age', 'Tm']
    #for col in df.columns:
    #    if col not in cat_columns + percentage_columns:
    #        df[col] = df[col].astype(int)
    
    print("Length after cleaning:", len(df))
    return df



df = clean_data()
print(df.head())
print(df.dtypes)
#the total number of rows is 763
#actual total rows is 735 after removing repeated headers
#We have 91 rows with nan values
#the columns with nan values are  ['FG%', '3P%', '2P%', 'eFG%', 'FT%']
#we need to replace these values with 0; this means player did not have scores

# Find rows with NaN values
#nan_rows = df[df.isna().any(axis=1)]
#print(nan_rows)
#columns_with_nan = df.columns[df.isna().any()]
#print(columns_with_nan)

#rows_with_mp_zero = df[df['MP'] == 0]
#print(rows_with_mp_zero)

#playersselected <- players |> 
#  filter(MP>0) |> filter(Pos == "G") |> 
#  select(Player, Team, Pos, MP, `FG%`, `3P%`, AST, TOV, PTS) |> 
 # na.omit() 