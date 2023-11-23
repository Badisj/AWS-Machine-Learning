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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score,
    accuracy_score
)

from datetime import datetime
from pathlib import Path
from pprint import pprint



#########################################################################
############################ Parse arguments ############################
def parse_args():
    parser = argparse.ArgumentParser(description='Process')
    
    # ====================== Model hyperparameters =====================
    parser.add_argument('--churn_month', type=int,
        default=1
    )
    
    parser.add_argument('--n_estimators', type=int,
        default=5
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
def load_data(path):
    input_files = glob.glob('{}/*.csv'.format(path))[0]
    print('Importing {}'.format(input_files))
    data_frame = pd.read_csv(input_files)
    
    data_frame = data_frame.select_dtypes([int,float])
    data_frame.rename(columns = {'churn_mon': 'churn_mon1'}, inplace = True)
    data_frame.info()
    
    return data_frame



########################################################################
########################### Models training ############################
def train_models(df_train, df_val, churn_month, n_estimators, max_depth, criterion, random_state):
    
    # ================== Retrieve features & targets ================
    print('Retrieving features & target: month {}'.format(churn_month))
    
    train_label = 'churn_mon{}'.format(churn_month)
    
    X_train = df_train.drop(columns=['churn_mon1', 'churn_mon2'])
    y_train = df_train[train_label]
    
    X_val = df_val.drop(columns=['churn_mon1', 'churn_mon2'])
    y_val = df_val[train_label]
    
    
    # ================== Instanciate and fit models =================
    print('Fitting models')
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 criterion=criterion,
                                 random_state=random_state)
    
    clf.fit(X_train, y_train)
    print('Model fitted.')
    
    
    # ================== Compute validation scores =================
    print('Evaluation: churn prediction month {}'.format(churn_month))
    pred_churn = clf.predict(X_val)
    pred_churn_proba = clf.predict_proba(X_val)[:, 1]
    
    precision = precision_score(y_val, pred_churn)
    recall = recall_score(y_val, pred_churn)
    f1 = f1_score(y_val, pred_churn)
    
    roc_auc = roc_auc_score(y_val, pred_churn_proba)
    accuracy = accuracy_score(y_val, pred_churn)
    print('val_precision: {0:.2f} - val_recall: {1:.2f} - val_f1score: {2:.2f} - val_roc_auc: {3:.2f} - val_accuracy: {4:.2f}%'.format(precision, 
                                                                                                                                       recall, 
                                                                                                                                       f1,
                                                                                                                                       roc_auc,
                                                                                                                                       100*accuracy))
    print('Training Complete.')
    
    
    # ========================= Save Models ========================
    path = os.path.join(args.model_dir, "clf_mon{}.joblib".format(churn_month))
    joblib.dump(clf, path)
    
    print('Model persisted at {}' + args.model_dir)
    
    return clf
    
    
########################################################################
################################ Main  #################################    
if __name__ == "__main__":
    args = parse_args()
    print('Loaded arguments:')
    print(args)
    
    data_train = load_data(args.train_data)
    data_valid = load_data(args.validation_data)
    
    clf_mon1 = train_models(df_train = data_train, 
                            df_val = data_valid,
                            churn_month=args.churn_month,
                            n_estimators = args.n_estimators, 
                            max_depth = args.max_depth, 
                            criterion=args.criterion, 
                            random_state=args.random_state)
    
    
    # Prepare for inference which will be used in deployment
    # You will need three files for it: inference.py, requirements.txt, config.json
    inference_path = os.path.join(args.model_dir, "code/")
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
    os.system("cp requirements.txt {}".format(inference_path))
    os.system("cp config.json {}".format(inference_path))
    
    
