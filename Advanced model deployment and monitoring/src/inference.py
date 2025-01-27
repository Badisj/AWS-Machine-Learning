import os 
import sys
import tarfile
import numpy as np
import pandas as pd

import joblib
from sklearn.ensemble import RandomForestClassifier

model_name_tar_gz = 'model.tar.gz'
model_name = 'churn_random_forest.joblib'




def model_fn(model_dir):
    # print('Extracting model.tar.gz')
    # model_tar_path = '{}/{}'.format(model_dir, model_name_tar_gz)
    # model_tar = tarfile.open(model_tar_path, 'r:gz')
    # model_tar.extractall(model_dir)
    
    print('Listing content of model dir: {}'.format(model_dir))
    model_files = os.listdir(model_dir)
    for mdl in model_files:
          print(mdl)
        
    model_path = os.path.join(model_dir, model_name)
    model = joblib.load(model_path)
    return model



def predict_fn(input_data, model):
    preds = model.predict(input_data)
    preds_proba = model.predict_proba(input_data)
    preds_df = np.concatenate((preds, preds_proba), axis=1)

    return preds_df


