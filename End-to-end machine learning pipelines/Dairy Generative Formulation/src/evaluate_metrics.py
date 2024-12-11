import subprocess
import argparse
import tarfile
import joblib
import pickle
import json
import glob
import sys
import os


subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib==3.2.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})


from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)



def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(',')


def pars_args():
    resconfig = {}
    try:
        with open('/opt/ml/config/resourceconfig.json', 'r') as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print('/opt/ml/config/resourceconfig.json not found.  current_host is unknown.')
        pass # Ignore
    
    
    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    # ========================= Parse arguments ========================
    parser.add_argument('--input-model', type=str,
        default='/opt/ml/processing/input/model',
    )
    
    
    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    
    
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    
    parser.add_argument('--sensory-target', type=str,
        default='sweetness',
    )
    
    parser.add_argument('--hosts', type=list_arg,
        default=resconfig.get('hosts', ['unknown']),
        help='Comma-separated list of host names running the job'
    )
    
    parser.add_argument('--current-host', type=str,
        default=resconfig.get('current_host', 'unknown'),
        help='Name of this host running the job'
    )

    return parser.parse_args()


    

def model_fn(model_name, model_dir):
    
    print('Extracting model.tar.gz')
    model_name_tar_gz = 'model.tar.gz'
    
    model_tar_path = '{}/{}'.format(model_dir, model_name_tar_gz)
    model_tar = tarfile.open(model_tar_path, 'r:gz')
    model_tar.extractall(model_dir)
    
    print('Listing content of model dir: {}'.format(model_dir))
    model_files = os.listdir(model_dir)
    for mdl in model_files:
          print(mdl)
        
    model_path = os.path.join(model_dir, model_name)
    model = joblib.load(model_path)
    
    return model



def predict_fn(input_data, model):
    preds = pd.Series(model.predict(input_data), name='prediction')
    return preds.values



def process(args):

    sensory_target = args.sensory_target

    print('Current host: {}'.format(args.current_host))
    print('Input data: {}'.format(args.input_data))
    print('Input model: {}'.format(args.input_model))
    
    
    # ========================== Model import =========================
    print('Start model import')
    
    model_name = "dairy_generative_formulation_{}.joblib".format(sensory_target)
    model = model_fn(
        model_name=model_name, 
        model_dir=args.input_model
    )
    
    # ========================== Data import ==========================
    print('Listing contents of input data dir: {}'.format(args.input_data))
    input_files = glob.glob('{}/*.csv'.format(args.input_data))
    
    print('Input files: {}'.format(input_files))
    
    try:
        df_test = pd.read_csv(input_files[0])
        for file in input_files[1:]:
            file_path = os.join(arsgs.input_data, file)
            df_temp = pd.read_csv(file_path, 
                                  engine='python',
                                  header=None)
            df_test = df_test.append(df_temp)
         
    except IndexError:
        df_test = pd.read_csv(input_files[0], 
                              engine='python',
                              header=None)
        
    print('Import input files: {} complete.'.format(input_files))


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
    print('Target: {}'.format(sensory_target))
    
    if sensory_target != 'Global Appreciation':
        X_test = df_test.drop(columns=sensory_attributes).copy()

    elif sensory_target == 'Global Appreciation':
        X_test= df_test.drop(columns=[sensory_target]).copy()

    y_true = df_test[sensory_target].values
    print('Testing data shape:', X_test.shape)


    # ====================== Generate predictions ======================
    print('Generating predictions')
    y_pred = predict_fn(X_test, model)
    

    # ==================== Compute metrics  ====================    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = mse**0.5
    
    print('predicted shape:', y_pred.shape)
    print('-------------------------- Evaluation metrics --------------------------')
    print('val_mse:', mse)
    print('val_rmse:', rmse)
    print('val_mae:', mae)
    print('val_r2:', r2)
    

    # ==================== Model outputs  ====================
    print('Saving outputs at {}'.format(args.output_data))
          
    metrics_path = os.path.join(args.output_data, 'metrics/')
    os.makedirs(metrics_path, exist_ok=True)
                
    report_dict = {
        "metrics": {
            "MSE": {
                "value": mse,
            },

            "RMSE": {
                "value": rmse,
            },

            "MAE": {
                "value": mae,
            },
    
            "R2": {
                "value": r2,
            },
        },
    }
          
    evaluation_path = '{}/evaluation_{}.json'.format(metrics_path, sensory_target)
    with open(evaluation_path, 'w') as f:
          f.write(json.dumps(report_dict))
      
          
    print('Listing content of output dir: {}'.format(args.output_data))
    output_files = os.listdir(args.output_data)
    for file in output_files:
          print(file)
          
    print('Listing content of metrics dir: {}'.format(metrics_path))
    metric_files = os.listdir(metrics_path)
    for file in metric_files:
          print(file)
          
    print('Model evaluation complete.')
          

        
################################################################################################################################################
#################################################################### Main ######################################################################
################################################################################################################################################    
    
if __name__ == "__main__":
    args = pars_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)