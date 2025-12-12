import pandas as pd
import os

mapping = {1 : 'yes', 0 : 'no'}

# for classification 
def LoadDataLikeability():
    column_names = ['age', 'plays_games', 'owns_cards', 'watches_anime', 'likes_pokemon']
    filepath = os.path.join(os.path.dirname(__file__), '../data/pokemon_likeability.csv')
    data = pd.read_csv(filepath, names=column_names)
    return data

# for continuous
def LoadDataPrice():
    column_names = ['rarity', 'age', 'condition', 'popularity', 'price']
    filepath = os.path.join(os.path.dirname(__file__), '../data/pokemon_prices.csv')
    data = pd.read_csv(filepath, names=column_names)
    return data

def LoadMapping():
    return mapping