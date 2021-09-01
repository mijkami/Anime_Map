from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, concatenate, Input, Dropout, Dense
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
from keras.layers import BatchNormalization
import pickle
from tensorflow.keras.callbacks import EarlyStopping


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-664-gogunska-neumf-perceptron'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/active_users_df.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'NeuMF_MLperceptron_full_data'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Optimize memory for lighter dataset
def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numerical columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return:
    """
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #to run in cloud
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    #to run locally - faster
    #df = pd.read_csv("data/processed_data/active_users_df.csv", nrows=5000000)
    norm_rating = df.rating/10
    df['rating'] = norm_rating
    df_optimized(df)
    print('optimized mem')
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

def train_model(num_users,num_animes,train):
    latent_dim = 10
    # Define inputs
    anime_input = Input(shape=[1],name='anime-input')
    user_input = Input(shape=[1], name='user-input')

    # MLP Embeddings
    anime_embedding_mlp = Embedding(num_animes + 1, latent_dim, name='anime-embedding-mlp')(anime_input)
    anime_vec_mlp = Flatten(name='flatten-anime-mlp')(anime_embedding_mlp)

    user_embedding_mlp = Embedding(num_users + 1, latent_dim, name='user-embedding-mlp')(user_input)
    user_vec_mlp = Flatten(name='flatten-user-mlp')(user_embedding_mlp)

    # MF Embeddings
    anime_embedding_mf = Embedding(num_animes + 1, latent_dim, name='anime-embedding-mf')(anime_input)
    anime_vec_mf = Flatten(name='flatten-anime-mf')(anime_embedding_mf)

    user_embedding_mf = Embedding(num_users + 1, latent_dim, name='user-embedding-mf')(user_input)
    user_vec_mf = Flatten(name='flatten-user-mf')(user_embedding_mf)

    concat = concatenate([anime_vec_mlp, user_vec_mlp], axis=1, name='concat')
    concat_dropout = Dropout(0.2)(concat)
    fc_1 = Dense(100, name='fc-1', activation='relu')(concat_dropout)
    fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
    fc_1_dropout = Dropout(0.2)(fc_1_bn)
    fc_2 = Dense(50, name='fc-2', activation='relu')(fc_1_dropout)
    fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
    fc_2_dropout = Dropout(0.2)(fc_2_bn)

    # Prediction from both layers
    pred_mlp = Dense(10, name='pred-mlp', activation='relu')(fc_2_dropout)
    pred_mf = dot([anime_vec_mf, user_vec_mf],
                axes=1,
                normalize=False,
                name='pred-mf')
    combine_mlp_mf = concatenate([pred_mf, pred_mlp],
                                axis=1,
                                name='combine-mlp-mf')

    # Final prediction
    result = Dense(1, name='result', activation='relu')(combine_mlp_mf)
    model = Model([user_input, anime_input], result)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    #es = EarlyStopping()
    model.compile(Adam(learning_rate=0.1), loss='mse', metrics='mse')
    model.fit([train.user_id, train.anime_id],
              train.rating,
              epochs=30,
              validation_split=0.3,
              verbose=0,
              callbacks=[es])
    print("trained model")
    return model


STORAGE_LOCATION = 'models/anime_map/NeuMF_MLperceptron_full_data_delllaptop_30epochs'


def save_model(model):
    # #tf.keras.models.save_model(model, 'NeuMF_MLperceptron_full_data.h5')
    # pickle.dump(model, open('NeuMF_MLperceptron_full_data', 'wb'))
    # print("saved model NeuMF_MLperceptron_full_data_1batch.pickle locally")
    # upload_model_to_gcp()
    # print(
    #     f"uploaded NeuMF_MLperceptron_full_data_1batch to gcp cloud storage under \n => {STORAGE_LOCATION}"
    # )
    tf.keras.models.save_model(
        model, 'NeuMF_MLperceptron_full_data_delllaptop_30epochs.h5')
    print("saved model NOT joblib locally")
    upload_model_to_gcp()
    print(
        f"uploaded NeuMF_MLperceptron_full_data_delllaptop_30epochs.h5 to gcp cloud storage under \n => {STORAGE_LOCATION}"
    )


def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename(
        'NeuMF_MLperceptron_full_data_delllaptop_30epochs.h5')


if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()
    # preprocess data
    dataset = id_transform(df)
    num_users,num_animes = len_to_num(dataset)
    # split
    train,test = split(dataset)
    # train model
    model = train_model(num_users, num_animes,train)
    save_model(model)
