import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_data (df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    - remove duplicates
    - remove NaNs
    """

    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    #clinical_data= ['Case', 'years_to_birth', 'gender', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status', 'outcome' ]

    print("✅ data cleaned and scaled")
    return df

def preprocess_features(X: pd.DataFrame) ->np.ndarray:
    """
    - transform cleaned dataset with MinMaxScaler for all lfq-intensities of proteins.
    -

    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X.iloc[:, 12:179] = scaler.fit_transform(X.iloc[:, 12:179])
    X= X.iloc[:, 12:179]
    
    return X
