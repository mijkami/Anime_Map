# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* anime_map/*.py

black:
	@black scripts/* anime_map/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr anime_map-*.dist-info
	@rm -fr anime_map.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)



# project id - replace with your GCP project id
PROJECT_ID=argon-depot-318913

# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-data-664-gogunska-neumf-perceptron

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1


BUCKET_TRAINING_FOLDER = 'trainings'

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME= NeuMF_perceptron
FILENAME=trainer
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=neuMF_perceptron_training_$(shell date +'%Y%m%d_%H%M%S')


set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="/home/ka_wagon/code/mijka/anime_map/data/processed_data/anime_map_data_animelist_100plus_PG.csv"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
# # @gsutil cp data/processed_data/active_users_df.csv gs://wagon-ml-my-bucket-name/data/active_users_df.csv
# @gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
	# @gsutil cp data/processed_data/anime_map_data_animelist_100plus_PG.csv gs://wagon-ml-my-bucket-name/data/anime_map_data_animelist_100plus_PG.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
	--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
	--package-path ${PACKAGE_NAME} \
	--module-name ${PACKAGE_NAME}.${FILENAME} \
	--python-version=${PYTHON_VERSION} \
	--runtime-version=${RUNTIME_VERSION} \
	--config config.yaml \
	--region ${REGION} \
	--stream-logs
