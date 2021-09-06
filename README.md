# Data analysis
- [Github URL](https://github.com/mijkami/anime_map) 
- Description: Anime recommendation system based on Machine-learning.
- Data Source: [Anime Recommendation DB 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020)
- Type of analysis:
  - data cleaning
  - modelisation:
    - a supervised model (KNN)
    - an unsupervised model (Deep learning / neural network)
  - [an API](https://github.com/mijkami/AnimeMap_API) to serve the predictions from the calculated model
  - [a website](https://github.com/mijkami/AnimeMap_front) to allow users to get prediction results from the API/model

Please document the project the better you can.

# Startup the project

Clone the project:
```bash
mkdir ~/code/anime_map && cd "$_"
git clone git@github.com:mijkami/anime_map.git
cd anime_map

```

Then add a raw_data directory (as it is not tracked by git):
```bash
mkdir raw_data

```


# Install

Go to `https://github.com/{group}/anime_map` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Create a virtual env for the project:
```bash
pyenv virtualenv anime_map
```

Tell pyenv that we want to use this virtual env for our project:
```bash
cd ~/code/mijkami/anime_map/
```

And run:
```bash
pyenv local anime_map
```



Install requirements:

```bash
make install_requirements
```
