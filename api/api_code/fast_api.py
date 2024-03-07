import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from io import BytesIO, StringIO
import json

# imports for preprocessing and model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler

from model import load_model
from preprocess import load_scaler
#from taxifare.ml_logic.registry import load_model

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model up front --> not sure yet if that will work
# app.state.model = load_model()

@app.post("/predict_uploaded_file")
def predict_uploaded_file(file: UploadFile = File(...)):
    contents = file.file.read() # Reading content of 'myfile' in bytes
    # print(contents)
    decoded_str = contents.decode('utf-8') # Decoding contents into str type
    # decoded_str = StringIO(contents.decode('utf-8')) # Alternative using StringIO
    df_json = json.loads(decoded_str) # Reading string and converting to json (dictionary)
    df = pd.DataFrame(df_json) # Reading dictionary and converting into dataframe
    # results = {
    #     "value": float(df["Identifier"][0])
    #     }
    # return results

    X_pred = df.drop(columns="Identifier")


    # Preprocess
    # Load scaler
    scaler = load_scaler(path = '/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_scalers',
                         file = 'MinMax_20240306-102844.joblib'
                        )

    X_pred_proc = scaler.transform(X_pred)

    # Predict data
    model = load_model()

    outcome_num = int(model.predict(X_pred_proc)[0])
    if outcome_num == 0:
        outcome = "good cancer"
        probability = round(float(model.predict_proba(X_pred_proc)[0][0]), 4)
    else:
        outcome = "bad cancer"
        probability = round(float(model.predict_proba(X_pred_proc)[0][1]), 4)

    return {
                "Outcome": outcome,
                "Probability": probability
    }


@app.get("/")
def root():
    return {'greeting': 'Hello'}
