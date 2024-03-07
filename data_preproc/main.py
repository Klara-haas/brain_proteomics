import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config; set_config(display = "diagram")
from ml_logic.data import clean_data, preprocess_proteins_age_gen, preprocess_proteins_all, synthetic_data
from sklearn.model_selection import train_test_split

def preprocess_age_gen():
    """
    - Query raw dataset from 'raw_data' folder
    - split the data into train (60 %), val (20 %), test (20 %)
    - create synthetic data using SMOTE on training set
    - process data (proteins, age and gender)
    - Store processed data
    -
    """
    print(" Preprocess proteins, age and gender ")

    # Retrieve data
    file_path=os.path.join(os.path.dirname(__file__), '..','raw_data','brain_proteomics_data.csv')
    data = pd.read_csv(file_path)

    #Process data
    data_clean= clean_data(data)


    # define X and y
    X = data_clean.drop(['Case', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status', 'outcome'], axis = 1)
    y= data_clean['outcome']

    # train-test/val split -- 35 %
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.35, stratify=y)
    print (f"✅ training data splitted. Size : {X_train.shape}")

    # test-val split -- 50 %
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test)
    print (f"✅ testing data splitted. Size: {X_test.shape} ")
    print (f"✅ validation data splitted. Size: {X_val.shape}")

    # produce synthetic data with SMOTE on training set
    X_train, y_train = synthetic_data(X_train, y_train)

    # preprocess X_train, X_test and X_val
    X_train, X_val, X_test = preprocess_proteins_age_gen(X_train, X_val, X_test)

    print("✅ preprocess() done of proteins, age and gender \n")

    return X_train, y_train, X_val, y_val, X_test, y_test

preprocess_age_gen()

def preprocess_all():
    """
    - Query raw dataset from 'raw_data' folder
    - process data (proteins, age, gender, radiation therapy, grade, mutation count, Percent.aneuploidy, IDH status = all relevant features)
    - Store processed data
    -
    """
    print("✅ Use case: preprocess proteins, and all relevant features: ")

    # Retrieve data
    file_path=os.path.join(os.path.dirname(__file__), '..','raw_data','brain_proteomics_data.csv')
    data = pd.read_csv(file_path)

    #Process data
    data_clean= clean_data(data)

    # define X and y
    X_preproc = preprocess_proteins_all(data)
    y= data_clean['outcome']

    print("✅ preprocess() done of proteins and all relevant features \n")
    return X_preproc, y

#X_preproc, y =preprocess_all()
