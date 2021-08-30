from google.cloud import storage
from anime_map.pipeline import model_knn_anime_map, PCA_vector
import numpy as np
import joblib
import pandas as pd

name = "rating_complete"
minimun_of_rating = 100
name_file = f'{name}_{minimun_of_rating}plus_PG'
BUCKET_NAME = 'wagon-data-664-le_mehaute'
STORAGE_LOCATION = f'anime_map_data/{name_file}_knn_model.joblib'  #for data.py .csv  and for trainer.py .joblib


def get_data():
    #df_users = pd.read_csv(f'../data/Processed_data/active_users_df_100PlusRatings_partial.csv',nrows=100) # local
    df_users = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/{name_file}.csv') #for google cloud
    return df_users


class Trainer:
    def __init__(self, pca_pivot):
        self.pca_pivot = pca_pivot
        
    def train(self):
        return model_knn_anime_map(self.pca_pivot)

def pivot_matrix(df):
    np_df = df.to_numpy()
    print('starting pivot with numpy')
    cols, col_pos = np.unique(np_df[:, 0], return_inverse=True)
    rows, row_pos = np.unique(np_df[:, 1], return_inverse=True)

    print('create cols and rows')
    
    pivot_table = np.zeros((len(rows), len(cols)), dtype=np_df.dtype)
    print('pivot_table full of zeros')
    pivot_table[row_pos, col_pos] = np_df[:, 2]
    print('pivot_table completed')
    
    return pivot_table

def upload_model_to_gcp():


    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename(f'{name_file}_knn_model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, f'{name_file}_knn_model.joblib')
    print(f"saved {name_file}_knn_model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded {name_file}_knn_model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

def save_vector(reg,new_df):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    STORAGE_LOCATION = f'anime_map_data/{new_df}.csv'
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    #joblib.dump(reg, f'{new_df}.joblib')
    reg.to_csv(f'{new_df}.csv', index=False)
    print(f"saved {new_df}.csv locally")

    # Implement here
    upload_model_to_gcp(new_df)
    print(f"uploaded {new_df}.csv to gcp cloud storage under \n => {STORAGE_LOCATION}")

if __name__ == '__main__':
    df = get_data()
    df = pivot_matrix(df)
    print('pivot_matrix step ok')
    print(df)
    df =  PCA_vector(df)
    save_vector(df,f'{name_file}_pivot_df')
    model = Trainer(df)
    print('Trainer step ok')
    model_knn = model.train()
    print('model.train step ok')
    save_model(model_knn)
    print('save_model ok')
