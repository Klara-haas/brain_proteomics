import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config
set_config(display = 'diagram')
from sklearn.compose import make_column_transformer, make_column_selector
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import time
import os


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

def preprocess_proteins_age_gen(X_train, X_val, X_test):
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

def synthetic_data_gen_age(X, y) -> np.array:
    """
    - Create training data with SMOTE.
    - n_samples = 200 -> total number of data points for each class
    - output will be X_train and y_train equally balanced 200 samples per class (1 and 0)
    """
    sampler = SMOTE(sampling_strategy={0: y.value_counts()[0]*2, 1: y.value_counts()[1]*2})
    X_train, y_train = sampler.fit_resample(X, y)

    print (f"✅ synthetic data created on training set. Size of new training set: {X_train.shape}")

    return X_train, y_train
