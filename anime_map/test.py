import pandas as pd
import numpy as np


def get_data():
    #df_users = pd.read_csv(f'../data/Processed_data/active_users_df_100PlusRatings_partial.csv',nrows=100) # local
    df_users = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/rating_complete_100plus_PG.csv') #for google cloud
    return df_users

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
    
    
    
if __name__ == '__main__':
    print('start')
    df = get_data()
    print('data load')
    #df = df_optimized(df)
    #print('df_optimized')
    df = pivot_matrix(df)
    print(df)