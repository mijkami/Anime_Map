from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import model_to_dot
from keras.layers import dot
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
#from keras.models import load_model
from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
#from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-664-gogunska-anime_map'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/active_users_df.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'anime_map_NeuMF'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=5000)
    return df

def id_transform(dataset):
    anime_id_to_new_id = dict()
    id = 1
    for index, row in dataset.iterrows():
        if anime_id_to_new_id.get(row['anime_id']) is None:
            anime_id_to_new_id[row['anime_id']] = id
            dataset.at[index, 'anime_id'] = id
            id += 1
        else:
            dataset.at[index, 'anime_id'] = anime_id_to_new_id.get(row['anime_id'])
    user_id_to_new_id = dict()
    id = 1
    for index, row in dataset.iterrows():
        if user_id_to_new_id.get(row['user_id']) is None:
            user_id_to_new_id[row['user_id']] = id
            dataset.at[index, 'user_id'] = id
            id += 1
        else:
            dataset.at[index, 'user_id'] = user_id_to_new_id.get(row['user_id'])
    return dataset

def len_to_num(dataset):
    num_users = len(dataset.user_id.unique())
    num_animes = len(dataset.anime_id.unique())
    return num_users,num_animes

def split(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    return train,test

def matrix_fact(num_users,num_animes):
    # Matrix factorization
    latent_dim = 10
    anime_input = Input(shape=[1],name='anime-input')
    anime_embedding = Embedding(num_animes + 1, latent_dim, name='anime-embedding')(anime_input)
    anime_vec = Flatten(name='anime-flatten')(anime_embedding)
    user_input = Input(shape=[1],name='user-input')
    user_embedding = Embedding(num_users + 1, latent_dim, name='user-embedding')(user_input)
    user_vec = Flatten(name='user-flatten')(user_embedding)
    from keras.layers import dot
    prod = dot([anime_vec, user_vec], axes=1, normalize=False)
    return anime_input,user_input,prod

def train_model(train,user_input, anime_input,prod):
    # Adding NN upon MF
    prod_dropout = Dropout(0.2)(prod)
    fc_1 = Dense(100, name='fc-1', activation='relu')(prod)
    fc_1_dropout = Dropout(0.2, name='fc-1-dropout')(fc_1)
    fc_2 = Dense(50, name='fc-2', activation='relu')(fc_1_dropout)
    fc_2_dropout = Dropout(0.2, name='fc-2-dropout')(fc_2)
    fc_3 = Dense(1, name='fc-3', activation='relu')(fc_2_dropout)
    model = Model([user_input, anime_input], fc_3)
    model.compile(optimizer=Adam(learning_rate=0.1), loss = 'mean_squared_error')
    model.fit([train.user_id, train.anime_id], train.rating, epochs=10,verbose=0)
    print("trained model")
    return model

STORAGE_LOCATION = 'models/anime_map/neuMFmodel'

# def save_model(model):
#     """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
#     HINTS : use joblib library and google-cloud-storage"""

#     # saving the trained model to disk is mandatory to then beeing able to upload it to storage
#     # Implement here
#     joblib.dump(model, 'neuMFmodel.joblib')
#     #joblib.dump(model, 'neuMFmodel.pkl')
#     print("saved neuMFmodel.joblib locally")

#     # Implement here
#     upload_model_to_gcp()
#     print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

def save_model_backup(model):
    tf.keras.models.save_model(model, 'neuMFmodel.h5')
    print("saved model NOT joblib locally")
    upload_model_to_gcp()
    print(f"uploaded neuMFmodel.h5 to gcp cloud storage under \n => {STORAGE_LOCATION}")

# def save_model_h5(model):
#     model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
#     #     print("saved model.h5 locally")


#joblib.dump('filename.pkl')


def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('neuMFmodel.h5')


if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()
    # preprocess data
    dataset = id_transform(df)
    num_users,num_animes = len_to_num(dataset)
    train,test = split(dataset)
    anime_input,user_input,prod = matrix_fact(num_users,num_animes)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    model = train_model(train, user_input, anime_input,prod)
    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model_backup(model)
    #save_model(model)
    #save_model_h5(model)
