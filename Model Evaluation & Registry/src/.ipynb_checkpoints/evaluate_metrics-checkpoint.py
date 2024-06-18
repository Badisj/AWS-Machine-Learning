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


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay


model_name_tar_gz = 'model.tar.gz'
model_name = 'churn_random_forest.joblib'

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
    
    
    parser.add_argument('--hosts', type=list_arg,
        default=resconfig.get('hosts', ['unknown']),
        help='Comma-separated list of host names running the job'
    )
    
    parser.add_argument('--current-host', type=str,
        default=resconfig.get('current_host', 'unknown'),
        help='Name of this host running the job'
    )

    return parser.parse_args()


    

def model_fn(model_dir):
    
    print('Extracting model.tar.gz')
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
    preds_proba = pd.DataFrame(model.predict_proba(input_data), columns = ['class0', 'class1'])
    
    preds_df = pd.concat((preds, preds_proba), axis=1)
    return preds_df.values



def process(args):
    print('Current host: {}'.format(args.current_host))
    
    print('Input data: {}'.format(args.input_data))
    print('Input model: {}'.format(args.input_model))
    
    
    # ========================== Model import =========================
    print('Start model import')
    model = model_fn(args.input_model)
    
    
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
        
    df_test = df_test.select_dtypes([int, float])
    df_test.drop(columns=df_test.columns[1], inplace=True)
    
    print('Import input files: {} complete.'.format(input_files))
          

    
    # ====================== Generate predictions ======================
    print('Generating predictions')
    test_label = df_test.columns[0]
    
    X_test = df_test.drop(columns=[test_label]).values
    y_true = df_test[test_label].values
    
    Y_pred = predict_fn(X_test, model)
    y_pred, y_pred_proba = Y_pred[:, 0], Y_pred[:, 1:]
    
    print('Predicted Labels and probabilities: {}'.format(Y_pred))
          
          

    # ==================== Classifications metrics  ====================
    print(classification_report(y_true=y_true, y_pred=y_pred))
    print(accuracy_score(y_true=y_true, y_pred=y_pred))
    
          
    #### Confusion matrix ####
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
          
    plt.figure(figsize=(20,7))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
   
    sns.heatmap(cm, 
                cbar=True, 
                cmap=plt.cm.Blues, 
                ax=ax1)
    plt.title('Confusion Matrix - Bank Churn Prediction')
    
    #### ROC AUC Curve ####
    RocCurveDisplay.from_predictions(y_true=y_true,
                                     y_pred=y_pred_proba[:,1], 
                                     ax=ax2)
          
    plt.show()
    
    
    # ==================== Model outputs  ====================
    print('Saving outputs at {}'.format(args.output_data))
          
    metrics_path = os.path.join(args.output_data, 'metrics/')
    os.makedirs(metrics_path, exist_ok=True)
    plt.savefig('{}/confusion_roc_auc.png'.format(metrics_path))
    
          
    dic_metrics = classification_report(y_true=y_true, 
                                        y_pred=y_pred, 
                                        output_dict=True)
          
    evaluation_path = '{}/evaluation.json'.format(metrics_path)
    with open(evaluation_path, 'w') as f:
          f.write(json.dumps(dic_metrics))
      
          
    print('Listing content of output dir: {}'.format(args.output_data))
    output_files = os.listdir(args.output_data)
    for file in output_files:
          print(file)
          
    print('Listing content of metrics dir: {}'.format(metrics_path))
    metric_files = os.listdir(metrics_path)
    for file in metric_files:
          print(file)
          
    print('Complete')
          

        
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