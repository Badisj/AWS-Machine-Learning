import os
import sys
import time
import json
import glob
import boto3
import logging
import argparse
import botocore
import functools
import subprocess
import multiprocessing

if __name__ == "__main__":
    os.system('pip install sagemaker==2.35.0')

import pickle
import sagemaker
import pandas as pd

from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_group import FeatureGroup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datetime import datetime
from pathlib import Path




#########################################################################
############ low-level service client of the boto3 session  #############

# config = botocore.config.Config()

# sm = boto3.client(service_name='sagemaker', 
#                   config=config)

# sm_runtime = boto3.client('sagemaker-runtime',
#                           config=config)

# sess = sagemaker.Session(sagemaker_client=sm,
#                          sagemaker_runtime_client=sm_runtime)

# bucket = sess.default_bucket()
# role = sagemaker.get_execution_role()
# region = sess.boto_region_name

# region = os.environ['AWS_DEFAULT_REGION']
# sts = boto3.Session(region_name=region).client(service_name='sts', region_name=region)
# iam = boto3.Session(region_name=region).client(service_name='iam', region_name=region)
# featurestore_runtime = boto3.Session(region_name=region).client(service_name='sagemaker-featurestore-runtime', region_name=region)
# sm = boto3.Session(region_name=region).client(service_name='sagemaker', region_name=region)

# caller_identity = sts.get_caller_identity()
# assumed_role_arn = caller_identity['Arn']
# assumed_role_name = assumed_role_arn.split('/')[-2]
# get_role_response = iam.get_role(RoleName=assumed_role_name) 
# role = get_role_response['Role']['Arn']
# bucket = sagemaker.Session().default_bucket()

# sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region), 
#                             sagemaker_client=sm,
#                             sagemaker_featurestore_runtime_client=featurestore_runtime)

config = botocore.config.Config(user_agent_extra='bedissj-1699438736259')


sm = boto3.client(service_name='sagemaker', 
                  config=config)

sm_runtime = boto3.client('sagemaker-runtime',
                          config=config)

featurestore_runtime = boto3.client(service_name='sagemaker-featurestore-runtime', 
                                    config=config)

sess = sagemaker.Session(sagemaker_client=sm,
                         sagemaker_runtime_client=sm_runtime,
                         sagemaker_featurestore_runtime_client=featurestore_runtime)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name



#########################################################################
#################### Wait for feature store creation ####################
def wait_for_feature_group_creation(feature_group):
    status = feature_group.describe().get('FeatureGroupStatus')
    while status == 'Creating':
        time.sleep(15)
        status = feature_group.describe().get('FeatureGroupStatus')
        print('Current status: {}'.format(status))
    if status != 'Created':
        print(f'Feature group {feature_group.name} failed to create')
    else:
        print(f'Feature group {feature_group.name} sucessfully created')



#########################################################################
##################### Cast Object columns to String #####################
def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame



#########################################################################
######################### Create Feature Store ##########################
def create_feature_group(feature_group_name, df_feature_definition, column_event_time, column_id, prefix=None):
    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sess
    )

    feature_group.load_feature_definitions(data_frame=df_feature_definition)

    feature_group.create(
        s3_uri=f's3://{bucket}/{prefix}',
        record_identifier_name=column_id,
        event_time_feature_name=column_event_time,
        role_arn=role,
        enable_online_store=False 
    )
    
    wait_for_feature_group_creation(feature_group)
    feature_group.describe()
    
    return feature_group



#########################################################################
####################### Data import & processing  #######################
def process(args):
    # ========================= Argument Variables =========================
    input_data = args.input_data
    output_data = args.output_data
    
    validation_ratio = args.validation_split_percentage
    test_ratio = args.test_split_percentage
    feature_group_name = args.feature_group_name
    prefix = args.feature_store_offline_prefix
    
    
    print("Local input data path: {}".format(input_data))
    print("Local output data path: {}".format(output_data))
    
    print("Validation ratio".format(validation_ratio))
    print("Test ratio: {}".format(test_ratio))
    print("Feature Group Name: {}".format(feature_group_name))
    print("Feature Group Prefix: {}".format(prefix))
          
    
    
    # ============================= Import data =============================
    # file = '{}/*.csv'.format(input_data)
    filename = glob.glob('{}/*.csv'.format(input_data))[0]
    print("Import data: {}...".format(filename))
    
    df = pd.read_csv(filename)
    df.rename(
        columns ={'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1':'churn_mon1',
                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2':'churn_mon2'},
        inplace = True
    )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    column_date = 'date'
    column_id = 'CLIENTNUM'

    df[column_date] = timestamp
    print("Import data: {} Completed".format(filename))
    
    
    # ====================== Encode categorical data =======================
    to_drop = [column_id, column_date]
    for cat in df.drop(columns = to_drop).select_dtypes('object').columns:
        enc = LabelEncoder()
        df[cat] = enc.fit_transform(df[cat])
        
        # output = open('AWS-machine-learning-orchestration/Data Processing & Feature Store{}/encoders/{}.pkl'.format(output_data, cat), 'wb')
        # pickle.dump(enc, output)
        # output.close()
    
    
    # ========================== Train test split ==========================
    print("Start train test split...")
    
    df_holdout, df_test = train_test_split(df, test_size=test_ratio)
    df_train, df_validation = train_test_split(df_holdout, test_size=validation_ratio)

    df_train['split_type'] = 'Train'
    df_validation['split_type'] = 'Validation'
    df_test['split_type'] = 'Test'
    
    print("Training data shape: {}".format(df_train.shape))
    print("Validation data shape: {}".format(df_validation.shape))
    print("Test data shape: {}".format(df_test.shape))
    
    print("Train test split Competed")

    

    # ======================= Cast objects to string =======================
    print("Start casting object columns to string")
    
    df_train = cast_object_to_string(data_frame=df_train)
    df_validation = cast_object_to_string(data_frame=df_validation)
    df_test = cast_object_to_string(data_frame=df_test)
    
    print("Casting object columns to string Completed")
    
    
    
    # ======================== Create Feature Group ========================
    print("Start Feature Group Creation")
    
    try:
        feature_group = create_feature_group(feature_group_name=feature_group_name , 
                                             df_feature_definition=df_train, 
                                             column_event_time=column_date, 
                                             column_id=column_id, 
                                             prefix='')

        print("Feature Group Created")


        #  ========================== Ingest Features ==========================
        print('Ingesting Features...')

        feature_group.ingest(data_frame=df_train,
                             max_workers=1,
                             wait=True)


        feature_group.ingest(data_frame=df_validation,
                             max_workers=1,
                             wait=True)


        feature_group.ingest(data_frame=df_test,
                             max_workers=1,
                             wait=True)

        offline_store_status = None
        while offline_store_status != 'Active':
            try:
                offline_store_status = feature_group.describe()['OfflineStoreStatus']['Status']
            except:
                pass
            print('Offline store status: {}'.format(offline_store_status))    
            time.sleep(15)

        print('Features Ingested.')
        
    except ResourceInUse:
        print('Feature Group already exists, passing this step.')
    
    
    # ========================== Write CSV files ==========================
    filename_without_extension = Path(Path(filename).stem).stem
    
    # Month 1
    train_data = '{}/train'.format(output_data)
    validation_data = '{}/validation'.format(output_data)
    test_data = '{}/test'.format(output_data)
    
    df_train = pd.concat((df_train['churn_mon1'], df_train.drop(columns = ['churn_mon1'])), axis=1)
    df_train.to_csv('{}/{}.csv'.format(train_data, filename_without_extension), index=False, header=False)
    
    df_validation = pd.concat((df_validation['churn_mon1'], df_validation.drop(columns = ['churn_mon1'])), axis=1)
    df_validation.to_csv('{}/{}.csv'.format(validation_data, filename_without_extension), index=False, header=False)
    
    df_test = pd.concat((df_test['churn_mon1'], df_test.drop(columns = ['churn_mon1'])), axis=1)
    df_test.to_csv('{}/{}.csv'.format(test_data, filename_without_extension), index=False, header=False)
        
    
    print('Data Processing Complete.')
    
    
    
    
#########################################################################
########################### Arguments Parser  ###########################
def parse_args():
    # =================== Local testing with CLI args ==================
    parser = argparse.ArgumentParser(description='Process')

    
    # ========================= Parse arguments ========================
#     parser.add_argument('--hosts', type=list_arg,
#         default=resconfig.get('hosts', ['unknown']),
#         help='Comma-separated list of host names running the job'
#     )
    
#     parser.add_argument('--current-host', type=str,
#         default=resconfig.get('current_host', 'unknown'),
#         help='Name of this host running the job'
#     )
    
    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    
    
    parser.add_argument('--validation-split-percentage', type=float,
        default=0.10,
    )
    
    parser.add_argument('--test-split-percentage', type=float,
        default=0.20,
    )
    
    parser.add_argument('--feature-store-offline-prefix', type=str,
        default=None,
    )
    
    parser.add_argument('--feature-group-name', type=str,
        default=None,
    ) 
    
    return parser.parse_args()



#########################################################################
################################# Main  #################################
if __name__ == "__main__":
    
    args = parse_args()
    print('Loaded arguments:')
    print(args)
    
    print('Environment variables:')
    print(os.environ)
    process(args)
    
#     num_cpus = multiprocessing.cpu_count()
#     print('num_cpus {}'.format(num_cpus))

#     p = multiprocessing.Pool(num_cpus)
#     p.map(process, args)