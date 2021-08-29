import pandas as pd
import joblib

def get_data(name_file):
    active_users_df_10PlusRatings_partial = pd.read_csv(f'../data/Processed_data/{name_file}.csv') # local
    #active_users_df_10PlusRatings_partial = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/{name_file}.csv') #for google cloud
    return active_users_df_10PlusRatings_partial

def get_anime():
    anime_df_relevant_PG = pd.read_csv('../data/Processed_data/anime_df_relevant_PG.csv') # local
    #anime_df_relevant_PG = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/anime_df_relevant_PG.csv') # for google cloud
    return anime_df_relevant_PG.rename(columns={'MAL_ID' : 'anime_id'})

def process_data(name_file):
    data_users_df = get_data(name_file)
    data_users_df['rating'] = data_users_df['rating']/10
    
    anime_df_relevant_PG = get_anime()
    anime_name_df = anime_df_relevant_PG[['anime_id','Name']]
    data_users_df_merge = data_users_df.merge(anime_name_df, on = 'anime_id', how='inner')
    pivot_df = data_users_df_merge.pivot_table(index='anime_id',columns='user_id',values='rating').fillna(0)
    
    anime_Genres_df = anime_df_relevant_PG[['anime_id','Genres']]
    anime_Genres_df_encoded = pd.concat(objs = [anime_Genres_df.drop(columns = 'Genres', axis =1), anime_Genres_df['Genres'].str.get_dummies(sep=", ")], axis = 1)
    anime_Genres_df_encoded = anime_Genres_df_encoded.set_index('anime_id')
    
    pivot_df = pivot_df.merge(anime_Genres_df_encoded, how='inner',left_index=True, right_index=True)
    anime_name_pivot_df = data_users_df_merge[['anime_id','Name']].drop_duplicates()
    anime_name_pivot_df = anime_name_pivot_df.sort_values('anime_id')
    anime_name_pivot_df = anime_name_pivot_df.reset_index().drop(columns = 'index')
    
    return pivot_df, anime_name_pivot_df




if __name__ == '__main__':
    name_file = "active_users_df_10PlusRatings_partial"
    pivot_df, anime_name_pivot_df = process_data(name_file)
    print('start')
    joblib.dump(pivot_df, '../data/pivot_df.joblib')
    joblib.dump(anime_name_pivot_df, '../data/anime_name_pivot_df.joblib')
    print('finish')
