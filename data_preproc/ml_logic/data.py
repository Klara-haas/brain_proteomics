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

    print("✅ data cleaned and scaled")
    return df

def preprocess_features(df: pd.DataFrame) ->np.ndarray:
    """
    - transform cleaned dataset with MinMaxScaler for all lfq-intensities of proteins.
    - MinMaxScaler for age
    - Oridnal Encoder for gender

    """
    preproc_numerical = make_pipeline(
    MinMaxScaler()
    )
    preproc_categorical = make_pipeline(
    OrdinalEncoder()
    )
    preproc_base = make_column_transformer(
    (preproc_categorical, ['gender']),
    (preproc_numerical, make_column_selector(dtype_include=["int64", "float64"]))
    )

    X_preproc = pd.DataFrame(preproc_base.fit_transform(df))

    return X_preproc
