import requests
import pandas as pd
from bs4 import BeautifulSoup


# URL of the webpage
url = 'https://www.basketball-reference.com/leagues/NBA_2024_per_game.html'
out_file = 'NBA_2024_per_game_stats.csv'

def data_extract(url, id='per_game_stats'):

    # Send a GET request to the webpage
    response = requests.get(url)
    # Check if the request was successful
    response.raise_for_status()  
    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table with player stats
    table = soup.find('table', {'id': id})
    # Convert the HTML table to a DataFrame
    df = pd.read_html(str(table))[0]
    df.to_csv('NBA_2024_per_game_stats.csv', index=False)
    return df
# Display the DataFrame
df = data_extract(url)
print(df.head())
#df.columns
df['Pos'].unique()