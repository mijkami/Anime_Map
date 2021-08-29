import pandas as pd
from google.cloud import storage
from anime_map.name import *


def get_data(name_file):
    #active_users_df_10PlusRatings_partial = pd.read_csv(f'../data/Processed_data/{name_file}.csv') # local
    df_users = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/{name_file}.csv') #for google cloud
    return df_users

def get_anime():
    #anime_df_relevant_PG = pd.read_csv('../data/Processed_data/anime_df_relevant_PG.csv') # local
    anime_df_relevant_PG = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/anime_df_relevant_PG.csv') # for google cloud
    return anime_df_relevant_PG.rename(columns={'MAL_ID' : 'anime_id'})

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
