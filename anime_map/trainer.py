from google.cloud import storage
from anime_map.data import process_data
from anime_map.pipeline import model_knn_anime_map
import joblib

BUCKET_NAME = 'wagon-data-664-le_mehaute'
name_file = 'active_users_df_10PlusRatings_partial'

class Trainer:
    def __init__(self, pca_pivot):
        self.pca_pivot = pca_pivot
        
    def train(self):
        return model_knn_anime_map(self.pca_pivot)


STORAGE_LOCATION = 'models/anime_map/knn_model.joblib'


def upload_model_to_gcp():


    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('knn_model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'knn_model.joblib')
    print("saved knn_model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded knn_model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")



if __name__ == '__main__':
    pivot_df, anime_name_pivot_df = process_data(name_file)
    print('process_data step ok')
    model = Trainer(pivot_df)
    print('Trainer step ok')
    model_knn = model.train()
    print('model.train() step ok')
    save_model(model_knn)
    print('save_model ok')
    print(sklearn.__version__)