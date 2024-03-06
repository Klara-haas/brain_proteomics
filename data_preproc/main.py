import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config; set_config(display = "diagram")
from ml_logic.data import clean_data, preprocess_proteins_age_gen, preprocess_proteins_all


def preprocess_age_gen():
    """
    - Query raw dataset from 'raw_data' folder
    - process data (proteins, age and gender)
    - Store processed data
    -
    """
    print("✅ Use case: preprocess proteins, age and gender: ")

    # Retrieve data
    file_path=os.path.join(os.path.dirname(__file__), '..','raw_data','brain_proteomics_data.csv')
    data = pd.read_csv(file_path)

    #Process data
    data_clean= clean_data(data)

    # define X and y
    X_preproc = preprocess_proteins_age_gen(data)
    y= data_clean['outcome']

    print("✅ preprocess() done of proteins, age and gender \n")
    return X_preproc, y

#X_preproc, y =preprocess_age_gen()

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
