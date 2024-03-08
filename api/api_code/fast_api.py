import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from io import BytesIO, StringIO
import json
from model import load_model
from preprocess import load_scaler, preprocess_input, clean_data

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


# Test function to check that api and website can communicate
@app.post("/predict_uploaded_file_test")
def predict_uploaded_file_test(file: UploadFile = File(...)):
    contents = file.file.read()
    print(type(contents))
    decoded_str = contents.decode('utf-8')
    print(type(decoded_str))
    print(decoded_str)
    #print("test")

    rows = decoded_str.split('\n')

    # Split each row into columns
    data = [row.split(',') for row in rows]

    # Convert the data into a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    results = {
        'first value': float(df["Identifier"][0])
        }
    return results


# Run prediction
@app.post("/predict_uploaded_file")
def predict_uploaded_file(file: UploadFile = File(...)):
    contents = file.file.read()
    print(type(contents))
    decoded_str = contents.decode('utf-8')
    print(type(decoded_str))
    print(decoded_str)

    rows = decoded_str.split('\n')

    # Split each row into columns
    data = [row.split(',') for row in rows]

    # Convert the data into a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.head(3))

    df = df.drop(columns="Identifier", axis = 1)


    # Preprocess
    X_pred = preprocess_input(df)

    # Predict data
    model = load_model(path ='/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_models',
                        file = 'sgd_model.pkl')

    outcome_num = int(model.predict(X_pred)[0])
    if outcome_num == 0:
        outcome = "good cancer"
        probability = round(float(model.predict_proba(X_pred)[0][0]), 4)
    else:
        outcome = "bad cancer"
        probability = round(float(model.predict_proba(X_pred)[0][1]), 4)

    return {
                "Outcome": outcome,
                "Probability": probability
    }









@app.get("/")
def root():
    return {'greeting': 'Hello'}



###############################################################################
# for some reason this code doesn't work anymore
# @app.post("/predict_uploaded_file")
# def predict_uploaded_file(file: UploadFile = File(...)):
#     contents = file.file.read() # Reading content of 'myfile' in bytes
#     #print(contents)
#     decoded_str = contents.decode('utf-8') # Decoding contents into str type
#     # decoded_str = StringIO(contents.decode('utf-8')) # Alternative using StringIO
#     df_json = json.loads(decoded_str) # Reading string and converting to json (dictionary)
#     df = pd.DataFrame(df_json) # Reading dictionary and converting into dataframe
#     # results = {
#     #     "value": float(df["Identifier"][0])
#     #     }
#     # return results

#     data = df.drop(columns="Identifier")


#     # Preprocess
#     X_pred = preprocess_input(data)


#     # Predict data
#     model = load_model(path ='/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_scalers',
#                         file = 'svc_model.pkl')

#     outcome_num = int(model.predict(X_pred)[0])
#     if outcome_num == 0:
#         outcome = "good cancer"
#         probability = round(float(model.predict_proba(X_pred)[0][0]), 4)
#     else:
#         outcome = "bad cancer"
#         probability = round(float(model.predict_proba(X_pred)[0][1]), 4)

#     return {
#                 "Outcome": outcome,
#                 "Probability": probability
#     }
