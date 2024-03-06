import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config; set_config(display = "diagram")
from ml_logic.data import clean_data, preprocess_features
from colorama import Fore, Style

def preprocess():
    """
    - Query raw dataset from 'raw_data' folder
    - process query data
    - Store processed data
    -
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Retrieve data
    file_path=os.path.join(os.path.dirname(__file__), '..','raw_data','brain_proteomics_data.csv')
    data = pd.read_csv(file_path)

    #Process data
    data_clean= clean_data(data)

    # define X and y
    X_preproc = preprocess_features(data)
    y= data_clean['outcome']

    print("✅ preprocess() done \n")
    return X_preproc, y

#X_preproc, y =preprocess()
