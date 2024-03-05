import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# imports for preprocessing and model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler

from model import load_model
#from taxifare.ml_logic.registry import load_model

# define functions because I don't know how to import them

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
app.state.model = load_model()

# http://127.0.0.1:8000/predict?path=/home/jana/code/jfschulz/project-brain-proteomics/raw_data&file=Glioma-clinic-TCGA-proteins-test-with-identifier.csv
@app.get("/predict")
def predict(
            path: str, #"/home/jana/code/jfschulz/project-brain-proteomics/raw_data"
            file: str, # "Glioma-clinic-TCGA-proteins-test-with-identifier.csv"
            ):
    """
    Make a prediction for every row in your dataset.
    Input needs to be a csv file with rows = samples and columns = proteins.
    The first row has to contain the protein names or any other identifier that will serve as a header.
    If your file has a sample identifier column, name this column "Identifier"
    """

    df = pd.read_csv(f"{path}/{file}", header=0)

    if 'Identifier' in df.columns:
        X_pred = df.drop(["Identifier"], axis = 1)
    else:
        X_pred = df

    model = app.state.model

    # X_pred_proc = preprocess_features(X_pred)

    # Predict data
    outcome = pd.DataFrame(model.predict(X_pred), columns=["Outcome"])
    prob = pd.DataFrame(model.predict_proba(X_pred), columns=["Probability_0", "Probability_1"])

    # Merge results into dataframe
    result = pd.merge(prob,outcome, left_index=True, right_index=True)

    return result


@app.get("/")
def root():
    return {'greeting': 'Hello'}
