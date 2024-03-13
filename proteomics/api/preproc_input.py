from joblib import dump
import time
import os
import pandas as pd
import numpy as np
from sklearn import set_config
import joblib

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

    #Process data
    data_clean= clean_data(data)

    # define X
    X = data_clean.drop(['Case', 'gender', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status'], axis = 1)
    P17_list = ['Syk_p', 'YAP_pS127_p', 'AR_p', 'ACC1_p', 'YAP_p',
        'HER3_pY1289_p', 'c-Kit_p', 'ACC_pS79_p', 'STAT3_pY705_p', 'DJ-1_p',
        '53BP1_p', 'p27_p', 'PDK1_p', 'S6_pS235_S236_p', 'PRDX1_p', 'Bax_p', 'IRS1_p']

    X_P17 = X[['years_to_birth'] + P17_list]

    scaler_path= os.path.join(os.path.dirname(__file__),'..','models','scaler_20240311-112633.joblib')
    preproc_scaler = joblib.load(scaler_path)
    # preprocess X_train, X_test and X_val
    X_pred= preproc_scaler.transform(X_P17)

    return X_pred
