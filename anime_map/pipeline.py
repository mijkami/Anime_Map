from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import joblib

#name = "10plus"

def model_knn_anime_map(pca_pivot):
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'auto')
    model_knn.fit(pca_pivot)
    name = "10plus"
    joblib.dump(model_knn, f'knn_{name}.joblib')