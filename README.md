# appliance_pred
A machine learning project to predict Appliance energy consumption in a low energy building in Belgium. 

deatils of the dataset can be found on [Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction).

# Dataset
`KAG_energydata_complete.csv` is the original Kaggle data.


# Notebook
The Jupyter notebook `notebook.ipynb` contains all the processes from data loading, EDA, and the trained regression model and the paramter tuning.

# Script
`train.py` is the complete code to load, train the model and save the model.

`predict.py` is for serving the model api in deployment.

`predict_test.py` is the predict script for model prediction.

`Pipfile` and `Pipfile.lock` are files to reproduce the environment.


# Docker
The `Dockerfile` contains how to containerize the code and environment to run the model.

To build the docker file, use the command to docker build this way:

`docker build -t energy-prediction .`

To run the model in docker:

`docker run -it -p 9696:9696 energy-prediction:latest`

Open another terminal and run:
`python3 predict_test.py`