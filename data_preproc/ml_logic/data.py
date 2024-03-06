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

def preprocess_proteins_age_gen(df: pd.DataFrame) ->np.ndarray:
    """
    - transform cleaned dataset with MinMaxScaler for all lfq-intensities of proteins.
    - MinMaxScaler for age
    - Oridnal Encoder for gender

    """
    prot_age_gen = df.drop(['Case', 'histological_type', 'race', 'ethnicity', 'radiation_therapy', 'Grade', 'Mutation.Count', 'Percent.aneuploidy', 'IDH.status', 'outcome'], axis = 1)
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

    X_preproc = preproc_base.fit_transform(prot_age_gen)

    return X_preproc

def preprocess_proteins_all(df: pd.DataFrame) ->np.ndarray:
    """
    - transform cleaned dataset with MinMaxScaler for all lfq-intensities of proteins.
    - MinMaxScaler for age, years of birth, mutation count.
    - Oridnal Encoder for gender, radiation therapy, grade, IDH status

    """
    prot_all = df.drop(['Case', 'histological_type', 'race', 'ethnicity','outcome'], axis = 1)

    preproc_numerical = make_pipeline(
    MinMaxScaler()
    )
    preproc_categorical = make_pipeline(
    OrdinalEncoder()
    )

    preproc_all = make_column_transformer(
    (preproc_categorical, ['gender', 'radiation_therapy','Grade', 'IDH.status']),
    (preproc_numerical, make_column_selector(dtype_include=["int64", "float64"]))
)

    X_preproc = preproc_all.fit_transform(prot_all)

    return X_preproc
