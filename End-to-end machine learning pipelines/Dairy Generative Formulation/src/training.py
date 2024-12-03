import os
import sys
import time
import json
import glob
import boto3
import joblib
import logging
import argparse
import botocore
import functools
import subprocess
import multiprocessing

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)


#########################################################################
############################ Parse arguments ############################
def parse_args():
    parser = argparse.ArgumentParser(description='Process')
    
    # ====================== Model hyperparameters =====================    
    parser.add_argument('--n_estimators', type=int,
        default=100
    )
    
    parser.add_argument('--max_depth', type=int,
        default=3
    )
    
    parser.add_argument('--criterion', type=str,
        default='gini'
    )
    
    parser.add_argument('--random_state', type=int,
        default=2024
    )

    parser.add_argument('--sensory_output', type=str,
        default='sweetness'
    )
    
    # ====================== Container environment =====================
    parser.add_argument('--hosts', 
                        type=list, 
                        default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--current_host', 
                        type=str, 
                        default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--model_dir', 
                        type=str, 
                        default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--train_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--validation_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_VALIDATION'])
        
    parser.add_argument('--output_dir', 
                        type=str, 
                        default=os.environ['SM_OUTPUT_DIR'])
    
    parser.add_argument('--num_gpus', 
                        type=int, 
                        default=os.environ['SM_NUM_GPUS'])
    
    # ======================= Debugger arguments ======================
    
    return parser.parse_args()



########################################################################
############################# Data loader ##############################
def create_list_input_files(path):
    input_files = glob.glob('{}/*.csv'.format(path))
    print(input_files)
    return input_files


def load_data(path):
    input_files = create_list_input_files(path)
    
    print('Importing {}'.format(input_files))
    for file in input_files:
        data = pd.read_csv(file, engine='python')
    
    data = data.select_dtypes([int, float])
    print('Data import complete.')    
    return data



########################################################################
########################### Models training ############################
def model_training(df_train, df_val, sensory_attribute, n_estimators, max_depth, criterion, random_state, model_dir):
    
    # ================== List sensory attributes ================
    sensory_attributes = [
    'Flavor_intensity ',
    'sweetness',
    'Fruit_intensity',
    'Chalkiness',
    'Color_intensity',
    'thickness',
    'Coating',
    'Global Appreciation'
    ]
    
    # ================== Retrieve features & targets ================
    print('Retrieving features and targets.')
    print('target: {}'.format(sensory_attribute))
    
    if sensory_attribute != 'Global Appreciation':
        X_train = df_train.drop(columns=sensory_attributes).copy()
        X_val = df_val.drop(columns=sensory_attributes).copy()

    elif sensory_attribute == 'Global Appreciation':
        X_train = df_train.drop(columns=[sensory_attribute]).copy()
        X_val = df_val.drop(columns=[sensory_attribute]).copy()


    y_train = df_train[sensory_attribute].copy()
    y_val = df_val[sensory_attribute].copy()

    print('Training data shape:', X_train.shape)
    print('Training target shape:', y_train.shape)
    print('Validation data shape:', X_val.shape)
    print('Validation target shape:', y_val.shape)
    
    
    # ================== Instanciate and fit models =================
    print('Fitting model...')
    mdl = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state
    )
    
    mdl.fit(X_train, y_train)
    print('Model Fitted.')
    
    
    # ================== Compute validation scores =================
    y_pred = mdl.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    rmse = mse**0.5
    
    print('predicted shape:', y_pred.shape)
    print('-------------------------- Evaluation metrics --------------------------')
    print('val_mse:', mse)
    print('val_rmse:', rmse)
    print('val_mae:', mae)
    print('val_r2:', r2)
    
    
    # ========================= Save Models ========================
    path = os.path.join(model_dir, "dairy_generative_formulation_{}.joblib".format(sensory_attribute))
    joblib.dump(mdl, path)
    print('Model persisted at: ' + model_dir)
    
    return mdl



if __name__ == '__main__':
    
    # Argument parser
    args = parse_args()
    print(args)
    
    
    # Data loading
    train_data = load_data(args.train_data)
    validation_data = load_data(args.validation_data)
    
    
    # Model training
    estimator = model_training(df_train=train_data, 
                               df_val=validation_data,
                               sensory_attribute=args.sensory_output,
                               n_estimators=args.n_estimators, 
                               max_depth=args.max_depth,
                               criterion=args.criterion, 
                               random_state=args.random_state, 
                               model_dir=args.model_dir)
    
    
    # Prepare for inference which will be used in deployment
    # You will need three files for it: inference.py, requirements.txt, config.json
    inference_path = os.path.join(args.model_dir, "code/")
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
    os.system("cp requirements.txt {}".format(inference_path))
    os.system("cp config.json {}".format(inference_path))