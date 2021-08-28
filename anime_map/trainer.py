from data import process_data
from pipeline import model_knn_anime_map

name_file = 'active_users_df_10PlusRatings_partial'

class Trainer:
    def __init__(self, pca_pivot):
        self.pca_pivot = pca_pivot
        
    def train(self):
        model_knn_anime_map(self.pca_pivot)

if __name__ == '__main__':
    pivot_df, anime_name_pivot_df = process_data(name_file)
    model = Trainer(pivot_df)
    model.train()
    print('here')