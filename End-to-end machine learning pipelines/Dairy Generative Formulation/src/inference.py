import os 
import sys
import tarfile
import joblib
import numpy as np
import pandas as pd


model_name_tar_gz = 'model.tar.gz'
model_name = 'churn_random_forest.joblib'



def model_fn(model_dir):
    print('Extracting model.tar.gz')
    model_name_tar_gz = 'model.tar.gz'
    
    model_tar_path = '{}/{}'.format(model_dir, model_name_tar_gz)
    model_tar = tarfile.open(model_tar_path, 'r:gz')
    model_tar.extractall(model_dir)
    
    print('Listing content of model dir: {}'.format(model_dir))
    model_files = os.listdir(model_dir)
    for mdl in model_files:
        model_path = os.path.join(model_dir, mdl)
        model = joblib.load(model_path)
    
    return model



def predict_fn(input_data, model):
    preds = pd.Series(model.predict(input_data), name='PredictedValue')
    
    return preds.values


