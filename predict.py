import os
from math import sqrt
import numpy as np
import joblib
import pandas as pd
from trying_NeuMF.trainer import get_data, id_transform, len_to_num, split
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
PATH_TO_LOCAL_MODEL = 'model.joblib'

BUCKET_NAME = 'wagon-data-664-gogunska-anime_map'
BUCKET_TRAIN_DATA_PATH = 'data/active_users_df.csv'


def get_test():
    df = get_data()
    # preprocess data
    dataset = id_transform(df)
    num_users,num_animes = len_to_num(dataset)
    train,test = split(dataset)
    return test

def load_model():
    model = tf.keras.models.load_model('neuMFmodel.h5')
    #model = joblib.load('neuMFmodel.joblib')
    return model

def predict(test,model):
    y_pred = np.round(model.predict([test.user_id, test.anime_id]), decimals=2)
    y_true = test.rating
    mean_absolute_error(y_true, y_pred)
    return y_true,y_pred

def evaluate_model(y_true, y_pred):
    MAE = round(mean_absolute_error(y_true, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y_true, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


if __name__ == '__main__':
    test = get_test()
    model = load_model()
    y_true,y_pred = predict(test,model)
    res = evaluate_model(y_true, y_pred)
    print(res)
