import pandas as pd
import numpy as np
import joblib
from google.cloud import storage

# take list users with more than 100 completed anime
users_list_test = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/rating_complete_100plus_PG.csv')[['user_id']].drop_duplicates()
# take list users with more than 100 rated anime
users_list_rating = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/animelist_100plus_PG.csv')

# make list users with more than 100 rated anime and 100 completed anime where we will select users
# to feed metric_model_anime. likes we are sure the users will work on the two model of recommendation
users_list_rating_test = users_list_test.merge(users_list_rating, on = 'user_id', how='inner')[['user_id']].drop_duplicates()

# take the anime name
animelist_relevant = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/anime_df_relevant_PG.csv')[['MAL_ID','Name']].rename(columns={'MAL_ID':'anime_id'})
# and just keep the uselful one
users_list_rating_name_test = animelist_relevant.merge(users_list_rating, on='anime_id', how='inner' )

# take the
anime_id_animelist_100plus = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/animelist_100plus_PG_anime_id_df.csv')
anime_id_animelist_100plus = anime_id_animelist_100plus.merge(animelist_relevant, on = 'anime_id', how='inner')

anime_id_rating_complete_100plus = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/rating_complete_100plus_PG_anime_id_df.csv')
anime_id_rating_complete_100plus = anime_id_rating_complete_100plus.merge(animelist_relevant, on = 'anime_id', how='inner')

# import the models and their pivot
storage_client = storage.Client()
bucket_name = 'wagon-data-664-le_mehaute'
model_bucket='anime_map_data/rating_complete_100plus_PG_knn_model.joblib'
model_local='local.joblib'

bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_bucket)
blob.download_to_filename(model_local)
model_rating_complete_100plus = joblib.load(model_local)
#model_rating_complete_100plus = joblib.load('gs://wagon-data-664-le_mehaute/anime_map_data/rating_complete_100plus_PG_knn_model.joblib')
pivot_rating_complete_100plus = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/rating_complete_100plus_PG_PCA_vector_df.csv')

model_bucket = 'anime_map_data/animelist_100plus_PG_knn_model.joblib'

bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_bucket)
blob.download_to_filename(model_local)
model_animelist_100plus = joblib.load(model_local)

#model_animelist_100plus = joblib.load('gs://wagon-data-664-le_mehaute/anime_map_data/animelist_100plus_PG_knn_model.joblib')
pivot_animelist_100plus = pd.read_csv('gs://wagon-data-664-le_mehaute/anime_map_data/animelist_100plus_PG_PCA_vector_df.csv')


def recomendation_rating_complete_100plus_pca(anime_name, nb_recomendation = 10):
    index_nb = anime_id_rating_complete_100plus.index[anime_id_rating_complete_100plus['Name'] == anime_name].tolist()[0]
    distances, indices = model_rating_complete_100plus.kneighbors(pivot_rating_complete_100plus.iloc[index_nb,:].values.reshape(1, -1), n_neighbors = nb_recomendation + 1)

    prediction = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            prediction.append([pivot_rating_complete_100plus.index[indices.flatten()[i]],0])
        else:
            prediction.append([pivot_rating_complete_100plus.index[indices.flatten()[i]],distances.flatten()[i]])
    return prediction

def recomendation_animelist_100plus_pca(anime_name, nb_recomendation = 10):
    index_nb = anime_id_animelist_100plus.index[anime_id_animelist_100plus['Name'] == anime_name].tolist()[0]
    distances, indices = model_animelist_100plus.kneighbors(pivot_animelist_100plus.iloc[index_nb,:].values.reshape(1, -1), n_neighbors = nb_recomendation + 1)

    prediction = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            prediction.append([pivot_animelist_100plus.index[indices.flatten()[i]],0])
        else:
            prediction.append([pivot_animelist_100plus.index[indices.flatten()[i]],distances.flatten()[i]])
    return prediction

def metric_model_anime(user_id = 33, model = 'notation', nb_recomendation = 5, best_anime=3):
    user = users_list_rating_name_test.query(f'user_id == {user_id}')[['anime_id','rating','Name']].sort_values(by=['rating'], ascending=False)
    user_best_anime = user.iloc[0:best_anime]
    # means notation on user
    list_notation = [i for i in users_list_rating_name_test.query(f'user_id=={user_id}').rating.tolist() if i!=0]
    means_note = np.mean(list_notation)
    user_best_anime = user_best_anime[user_best_anime['rating'] >means_note]
    print(f'means_note : {means_note}')
    
    # init the value of the metric
    metric = []
    
    # chose the model
    list_anime = []
    for i in range(user_best_anime.shape[0]):
        list_anime.append(user_best_anime.iloc[[i][0]].Name)

        if model == 'notation':
            list_reco_vote = []
            # call the models for the different user_best_anime
            for anime_name in list_anime:
                list_reco_vote.append(recomendation_animelist_100plus_pca(anime_name, nb_recomendation)) 

        elif model == 'completed':
            list_reco_vote = []
            # call the models for the different user_best_anime
            for anime_name in list_anime:
                list_reco_vote.append(recomendation_rating_complete_100plus_pca(anime_name, nb_recomendation))

        else:
            print('bye')
            
    # loop on the predict for multiple anime
    max_score_list = []
    verif_score_list = []
    for i in range(len(list_reco_vote)):
        # detremine the max possible score if every pred is liked
        max_score = len(list_reco_vote[i][1:])
        # init verif_score for comparaison to max_score
        verif_score = 0
        # loop on the predict for one anime
        for j in range((len(list_reco_vote[i][1:]))):
        
            anime_id = list_reco_vote[i][j+1][0]
            rating = users_list_rating_name_test.query(f'anime_id=={anime_id}').query(f'user_id=={user_id}').rating.tolist()
            # test if a rating exist for the anime recommended
            if  rating != []:
                # take the rating for comparaison
                note = rating[0]
                # comparaison to means_note to know if liked or not
                if note >=means_note:
                    # liked 
                    verif_score +=1
                # if note == zero => no vote from users so max_score-1
                elif note == 0:
                    max_score -=1
                #print(note)
            # if rating == [] users did not see the anime so max_score-1
            elif rating == []:
                max_score -=1
        if max_score!=0:
            max_score_list.append(max_score)
            verif_score_list.append(verif_score)
            efficiancy = 100 * verif_score/max_score
            #print(f'efficiancy partial = {efficiancy}%')
            metric.append(efficiancy)
        else:
            #print('nothing')
            pass
    #print(f'efficiancy : {np.mean(metric)}%')
    #if np.mean(max_score_list)!= 0:
        #print(f'{100 * np.mean(verif_score_list)/np.mean(max_score_list)}%')
    return verif_score_list, max_score_list

if __name__ == '__main__':
    sample = users_list_rating_test.sample(n=50)
    print(f'sample shape : {sample.shape}')
    vsl_nota = []
    msl_nota = []
    vsl_comp = []
    msl_comp = []
    for id in sample.user_id.tolist():
        print(f'for user id : {id}')
        print(f'notation')
        vsl, msl = metric_model_anime(user_id=id,best_anime=5)
        vsl_nota.append(vsl)
        msl_nota.append(msl)
        print(f'completed')
        vsl, msl = metric_model_anime(user_id=id,model = 'completed',best_anime=5)
        vsl_comp.append(vsl)
        msl_comp.append(msl)
    vsl_nota_tot = 0
    msl_nota_tot = 0
    for i in range(len(vsl_nota)):
        for j in range(len(vsl_nota[i])):
            vsl_nota_tot +=vsl_nota[i][j]
    for i in range(len(msl_nota)):
        for j in range(len(msl_nota[i])):
            msl_nota_tot +=msl_nota[i][j]
    vsl_comp_tot = 0
    msl_comp_tot = 0
    for i in range(len(vsl_comp)):
        for j in range(len(vsl_comp[i])):
            vsl_comp_tot +=vsl_comp[i][j]
    for i in range(len(msl_comp)):
        for j in range(len(msl_comp[i])):
            msl_comp_tot +=msl_comp[i][j]
    print(f'score notation :{vsl_nota_tot/msl_nota_tot}')
    print(f'score completed :{vsl_comp_tot/msl_comp_tot}')
