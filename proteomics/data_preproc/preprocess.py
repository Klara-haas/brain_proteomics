from joblib import dump, load
import time
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn import set_config
set_config(display = 'diagram')
from sklearn.compose import make_column_transformer, make_column_selector



def clean_data (df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    - remove duplicates
    - remove NaNs
    """

    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    #clinical_data= ['Case', 'years_to_birth', 'gender', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status', 'outcome' ]

    print("✅ Data cleaned")
    return df

# Load scaler


def load_scaler(path = '~/api/saved_scalers',
               file = 'scaler.gz'
              ):
    """
    Loads a locally saved and fitted scaler from the given path and file name.
    """
    current_path = os.path.dirname(__file__)
    path_file = os.path.join(current_path,'..','models','scaler')

    scaler = load(path_file)
    return scaler


def preprocess_input(data) -> np.array:
    """
    - process data (proteins, age and gender)
    - load fitted scaler and pipline for preprocessing
    - return processed data
    """
    print(" Preprocess input proteins, age and gender... ")

    # Clean data
    data_clean= clean_data(data)

    # define X
    X = data_clean.drop(['Case', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status'], axis = 1)
    #X = data_clean
    current_path = os.path.dirname(__file__)
    preproc_scaler = load_scaler(path=os.path.join(current_path,'..','models'), file='scaler.joblib')
    # preprocess X_train, X_test and X_val
    X_pred= preproc_scaler.transform(X)

    return X_pred




def save_scaler(scaler_to_save = None,
               scaler_type = None,
               path_to_save = "/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_scalers"
              ):
    """
    Persist trained model locally on the hard drive at f"{path_to_save/scaler_type/f"{timestamp}.joblib"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save scaler locally
    scaler_path_file = os.path.join(f"{path_to_save}/{scaler_type}_{timestamp}.joblib")

    dump(scaler_to_save, scaler_path_file)

    print(f"✅ Scaler saved locally at {scaler_path_file}")

data_check=pd.read_csv('../../raw_data/Glioma-clinic-TCGA-proteins-all-cols-test-with-identifier-outcome1.csv')
#print(preprocess_input(data_check))

current_path = os.path.dirname(__file__)
preproc_scaler = load_scaler(path=os.path.join(current_path,'..','models'), file='scaler.joblib')
preproc_scaler.
