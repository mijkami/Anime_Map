import pandas as pd
import numpy as np


def get_data(name_file):
    active_users_df_10PlusRatings_partial = pd.read_csv(f'../data/Processed_data/{name_file}.csv')
    return active_users_df_10PlusRatings_partial

def get_anime():
    anime_df_relevant_PG = pd.read_csv('../data/Processed_data/anime_df_relevant_PG.csv')
    return anime_df_relevant_PG[['MAL_ID','Name']].rename(columns={'MAL_ID' : 'anime_id'})

def process_data(name_file):
    data_users_df = get_data(name_file)
    data_users_df['rating'] = data_users_df['rating']/10
    
    anime_df_relevant_PG = get_anime()
    anime_name_df = anime_df_relevant_PG[['MAL_ID','Name']].rename(columns={'MAL_ID' : 'anime_id'})
    data_users_df_merge = data_users_df.merge(anime_name_df, on = 'anime_id', how='inner')
    pivot_df = data_users_df_merge.pivot_table(index='anime_id',columns='user_id',values='rating').fillna(0)
    
    anime_Genres_df = anime_df_relevant_PG[['MAL_ID','Genres']].rename(columns={'MAL_ID' : 'anime_id'})
    anime_Genres_df_encoded = pd.concat([anime_Genres_df.drop('Genres', 1), anime_Genres_df['Genres'].str.get_dummies(sep=", ")], 1)
    
    pivot_df = pivot_df.merge(anime_Genres_df_encoded, how='inner',left_index=True, right_index=True)
    anime_name_pivot_df = data_users_df_merge[['anime_id','Name']].drop_duplicates()
    anime_name_pivot_df = anime_name_pivot_df.sort_values('anime_id')
    anime_name_pivot_df = anime_name_pivot_df.reset_index().drop(columns = 'index')
    
    return pivot_df, anime_name_pivot_df
