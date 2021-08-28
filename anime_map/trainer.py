from data import process_data
from pipeline import model_knn_anime_map

name_file = 

class Trainer:
    def __init__(self,pca_pivot)
        self.pca_pivot = pca_pivot
        
    def tain(self):
        model_knn_anime_map(pca_pivot)

if __name__ == '__main__':
<<<<<<< HEAD
    pivot_df, anime_name_pivot_df = process_data(name_file)
    model = Trainer(pivot_df)
    model.train()
    print('here')
=======
    pivot_df, anime_name_pivot_df = process_data('animelist')
    
>>>>>>> parent of a5cdf8e... update 27_08
