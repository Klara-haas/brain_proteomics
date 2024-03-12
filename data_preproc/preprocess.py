from joblib import dump
import time
import os
import pandas as pd
import numpy as np
from sklearn import set_config
from data import clean_data, scale_proteins, synthetic_data
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def clean_data (df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    - remove duplicates
    - remove NaNs
    """
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    print("✅ Data cleaned")
    return df

def preprocess_input(data) -> np.array:

    print(" Preprocess input proteins, age and gender... ")

    # Retrieve data
    file_path=os.path.join(os.path.dirname(__file__), '..','raw_data','brain_proteomics_data_input.csv')
    data = pd.read_csv(file_path)

    #Process data
    data_clean= clean_data(data)

    # define X
    X = data_clean.drop(['Case', 'gender', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status', 'outcome'], axis = 1)
    P17_list = ['Syk_p', 'YAP_pS127_p', 'AR_p', 'ACC1_p', 'YAP_p',
        'HER3_pY1289_p', 'c-Kit_p', 'ACC_pS79_p', 'STAT3_pY705_p', 'DJ-1_p',
        '53BP1_p', 'p27_p', 'PDK1_p', 'S6_pS235_S236_p', 'PRDX1_p', 'Bax_p', 'IRS1_p']

    X_P17 = X[['years_to_birth'] + P17_list]

    scaler_path= os.path.join(os.path.dirname(__file__),'..','models','scaler_20240311-112633.joblib')
    preproc_scaler = joblib.load(scaler_path)
    # preprocess X_train, X_test and X_val
    X_pred= preproc_scaler.transform(X_P17)

    return X_pred

def save_scaler(scaler_to_save = None,
               scaler_type = None,
               path_to_save = "..."
              ):
    """
    Save fitted scaler locally.
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save scaler locally
    scaler_path_file = os.path.join(f"{path_to_save}/{scaler_type}_{timestamp}.joblib")

    dump(scaler_to_save, scaler_path_file)

    print(f"✅ Scaler saved locally at {scaler_path_file}")

def preprocess_data():
    """
    - Query raw dataset from 'raw_data' folder
    - split the data into train (60 %), val (20 %), test (20 %)
    - create synthetic data using SMOTE on training set
    - process data (proteins, age and gender)
    - Store processed data
    -
    """
    print(" Preprocess proteins ... ")

    # Retrieve data
    file_path=os.path.join(os.path.dirname(__file__), '..','raw_data','brain_proteomics_data.csv')
    data = pd.read_csv(file_path)

    #Process data
    data_clean= clean_data(data)

    # define X and y
    X = data_clean.drop(['Case', 'gender', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status', 'outcome'], axis = 1)
    y= data_clean['outcome']

    # define 17 proteins that are used for model
    P17_list = ['Syk_p', 'YAP_pS127_p', 'AR_p', 'ACC1_p', 'YAP_p',
        'HER3_pY1289_p', 'c-Kit_p', 'ACC_pS79_p', 'STAT3_pY705_p', 'DJ-1_p',
        '53BP1_p', 'p27_p', 'PDK1_p', 'S6_pS235_S236_p', 'PRDX1_p', 'Bax_p', 'IRS1_p']

    X_P17 = X[['years_to_birth'] + P17_list]

    # train-test/val split -- 35 %
    X_train, X_val_test, y_train, y_val_test = train_test_split(X_P17, y, test_size=0.35, stratify=y)
    print (f"✅ Training data splitted. Size : {X_train.shape}")

    # test-val split -- 50 %
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test)
    print (f"✅ Testing data splitted. Size: {X_test.shape} ")
    print (f"✅ Validation data splitted. Size: {X_val.shape}")

    # produce synthetic data with SMOTE on training set
    X_train, y_train = synthetic_data(X_train, y_train)

    # preprocess X_train, X_test and X_val
    X_train, X_val, X_test, preproc_base = scale_proteins(X_train, X_val, X_test)

    print("✅ Preprocess done of proteins, age and gender \n")

    return X_train, y_train, X_val, y_val, X_test, y_test, preproc_base

def synthetic_data(X, y) -> np.array:
    """
    - Create training data with SMOTE.
    - n_samples = 200 -> total number of data points for each class
    - output will be X_train and y_train equally balanced 200 samples per class (1 and 0)
    """
    sampler = SMOTE(sampling_strategy={0: y.value_counts()[0]*2, 1: y.value_counts()[1]*2})
    X_train, y_train = sampler.fit_resample(X, y)

    print (f"✅ synthetic data created on training set. Size of new training set: {X_train.shape}")

    return X_train, y_train

def scale_proteins(X_train, X_val, X_test):
    """
    - transform cleaned dataset with MinMaxScaler for all lfq-intensities of proteins.
    - MinMaxScaler for age
    - Oridnal Encoder for gender

    """
    preprocessor = MinMaxScaler()

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test= preprocessor.transform(X_test)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path_to_save = os.path.join(os.path.dirname(__file__), '..','models/')
    scaler_path_file = os.path.join(f"{path_to_save}/scaler_{timestamp}.joblib")

    dump(preprocessor, scaler_path_file)

    return X_train, X_val, X_test, preprocessor
