import pandas as pd
import numpy as np


def get_data():
    #df_users = pd.read_csv(f'../data/Processed_data/active_users_df_100PlusRatings_partial.csv',nrows=100) # local
    df_users = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/rating_complete_100plus_PG.csv') #for google cloud
    return df_users

def get_anime():
    #anime_df_relevant_PG = pd.read_csv('../data/Processed_data/anime_df_relevant_PG.csv') # local
    anime_df_relevant_PG = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/anime_df_relevant_PG.csv') # for google cloud
    return anime_df_relevant_PG.rename(columns={'MAL_ID' : 'anime_id'})

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
    
def anime_genres_encoded(df):
    print('pandas version:')
    print(pd.__version__)
    #all anime_id on df
    anime_id_df =  df.sort_values(by=['anime_id']).drop_duplicates('anime_id').reset_index()[['anime_id']]
    # genres by anime_id
    anime_genre_df = get_anime()[['anime_id','Genres']]
    #just gor anime_id in this df
    anime_genre_df = anime_genre_df.merge(anime_id_df, on='anime_id',how='inner')
    #one hot encod genres for anime_genre_df
    anime_genres_df_encoded = pd.concat(objs = [anime_genre_df.drop(columns = 'Genres', axis =1), anime_genre_df['Genres'].str.get_dummies(sep=", ")], axis = 1)
    anime_genres_df_encoded = anime_genres_df_encoded.set_index('anime_id')
    #convert to numpy array
    anime_genres_np = anime_genres_df_encoded.to_numpy()
    
    return anime_genres_np
    
if __name__ == '__main__':
    print('start')
    df = get_data()
    print('data load')
    df = anime_genres_encoded(df)
    print('data load')
    #df = df_optimized(df)
    #print('df_optimized')
    #df = pivot_matrix(df)
    print(df)