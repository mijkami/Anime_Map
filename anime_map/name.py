
name_file = "rating_complete"
minimun_of_rating = 10
new_df = f'{name_file}_{minimun_of_rating}plus_PG'
BUCKET_NAME = 'wagon-data-664-le_mehaute'
STORAGE_LOCATION = f'anime_map_data/{new_df}.joblib'  #for data.py .csv  and for trainer.py .joblib
