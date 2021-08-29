import pandas as pd
from google.cloud import storage

name_file = "rating_complete"
minimun_of_rating = 10
new_df = f'{name_file}_{minimun_of_rating}plus_PG'
BUCKET_NAME = 'wagon-data-664-le_mehaute'
STORAGE_LOCATION = f'anime_map_data/{new_df}.csv'

def get_data(name_file):
    #active_users_df_10PlusRatings_partial = pd.read_csv(f'../data/Processed_data/{name_file}.csv') # local
    df_users = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/{name_file}.csv') #for google cloud
    return df_users

def get_anime():
    #anime_df_relevant_PG = pd.read_csv('../data/Processed_data/anime_df_relevant_PG.csv') # local
    anime_df_relevant_PG = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/anime_df_relevant_PG.csv') # for google cloud
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

def create_data(name_file):
    df_vote = get_data(name_file)
    df_vote = df_vote[['user_id', 'anime_id', 'rating']]
    print(df_vote.shape)
    df_id_PG = get_anime()[['anime_id']]
    df_vote_PG = df_vote.merge(df_id_PG, on = 'anime_id', how='inner')
    print(df_vote_PG.shape)
    df_vote_PG = df_vote_PG[df_vote_PG.rating !=0]
    print(df_vote_PG.shape)
    counts = df_vote_PG['user_id'].value_counts()
    active_users_df = df_vote_PG[df_vote_PG['user_id'].isin(counts[counts >= minimun_of_rating].index)]
    
    return active_users_df


def modif_data(name_file):
    df = get_data(name_file)
    print(df.columns)
    df['rating'] = 1
    df.rename(columns={'rating' : 'complete'}, inplace =True)
    
    return df
    
def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename(f'{new_df}.csv')


def save_data(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    #joblib.dump(reg, f'{new_df}.joblib')
    reg.to_csv(f'{new_df}.csv', index=False)
    print(f"saved {new_df}.csv locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded {new_df}.csv to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    active_users_df = modif_data(new_df)
    print(active_users_df.shape)
    save_data(active_users_df)
    print('finish')
