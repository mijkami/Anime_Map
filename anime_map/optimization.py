import pandas as pd
from google.cloud import storage

BUCKET_NAME = 'wagon-data-664-le_mehaute'
name_file = 'animelist_10plus_PG'
STORAGE_LOCATION = f'anime_map_data/{name_file}.csv'  #for data.py .csv  and for trainer.py .joblib

def get_data():
    #df_users = pd.read_csv(f'../data/Processed_data/active_users_df_100PlusRatings_partial.csv',nrows=100) # local
    df_users = pd.read_csv(f'gs://wagon-data-664-le_mehaute/anime_map_data/{name_file}.csv') #for google cloud
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


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename(f'{name_file}.csv')


def save_data(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    #joblib.dump(reg, f'{new_df}.joblib')
    reg.to_csv(f'{name_file}.csv', index=False)
    print(f"saved {name_file}.csv locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded {name_file}.csv to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    print('start')
    df = get_data()
    print('get_data ok')
    df = df_optimized(df)
    print('df_optimized ok')
    save_data(df)
    print('finish')
