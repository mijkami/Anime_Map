from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from anime_map.data import get_anime
from google.cloud import storage
import pandas as pd
import numpy as np

name_file = "rating_complete"
minimun_of_rating = 10
df_to_vect = f'{name_file}_{minimun_of_rating}plus_PG'
BUCKET_NAME = 'wagon-data-664-le_mehaute'

def normalisation_data(df):
    # uniquely for the animelist_10plus and animelist_100plus
    df['rating'] = df['rating']/10
    return df

'''
def vectorisation_data(df):

    anime_df_relevant_PG = get_anime()
    anime_id = anime_df_relevant_PG[['anime_id']]
    anime_Genres_df = anime_df_relevant_PG[['anime_id','Genres']]
    anime_Genres_df_encoded = pd.concat(objs = [anime_Genres_df.drop(columns = 'Genres', axis =1), anime_Genres_df['Genres'].str.get_dummies(sep=", ")], axis = 1)
    anime_Genres_df_encoded = anime_Genres_df_encoded.set_index('anime_id')
    print('anime_Genres_df_encoded ok')
    
    print(f'df_shape : {df.shape}')
    df_ = df.iloc[0:25*10**6,:]
    if 'rating' in df.columns:
        pivot_df = df_.pivot_table(index='anime_id',columns='user_id',values='rating').fillna(0)
    else:
        pivot_df = df_.pivot_table(index='anime_id',columns='user_id',values='complete').fillna(0)
    
    print('pivot_df ok')
    
    pivot_df = pivot_df.merge(anime_Genres_df_encoded, how='inner',left_index=True, right_index=True)
    anime_name_pivot_df = anime_df_relevant_PG[['anime_id','Name']].drop_duplicates().set_index('anime_id')
    anime_name_pivot_df = anime_name_pivot_df.merge(pivot_df[[]], how='inner',left_index=True, right_index=True)
    anime_name_pivot_df = anime_name_pivot_df.reset_index().sort_values('anime_id')
    
    return pivot_df, anime_name_pivot_df
'''

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

def PCA_vector(pivot_df):
    pca = PCA(n_components = 2000, svd_solver='full')
    pca.fit(pivot_df)
    print(pca.explained_variance_ratio_.sum())
    pca_array = pca.transform(pivot_df)
    return pd.DataFrame(pca_array)

def model_knn_anime_map(pca_pivot):
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'auto')
    model_knn.fit(pca_pivot)
    return model_knn

def upload_model_to_gcp(new_df):
    STORAGE_LOCATION = f'anime_map_data/{new_df}.csv'
    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename(f'{new_df}.csv')


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
    print('start')
    df = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/{df_to_vect}.csv')
    print('read_csv ok')
    pivot_df = pivot_matrix(df)
    print(f'pivot_df_shape : {pivot_df.shape}')
    save_vector(pivot_df, f'{df_to_vect}_pivot_df')
    pca_vector = PCA_vector(pivot_df)
    print(f'pca_vector_shape : {pca_vector.shape}')
    save_vector(pca_vector,  f'{df_to_vect}_pca_vector')
    print('finish')
