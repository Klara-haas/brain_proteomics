import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from joblib import dump, load
from io import BytesIO, StringIO
import json
from proteomics.data_preproc.preprocess import load_scaler, preprocess_input, clean_data
import os


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Test function to check that api and website can communicate
@app.post("/predict_uploaded_file_test")
def predict_uploaded_file_test(file: UploadFile = File(...)):
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

    results = {
        'first value': float(df["Identifier"][0])
        }
    return results


# Run prediction for one sample
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

    df = df.drop(columns="Identifier", axis = 1)
    # Preprocess
    X_pred = preprocess_input(df)

    # Predict data+
    current_path = os.path.dirname(__file__)
    model = load(os.path.join(current_path,'..','models', 'log_reg_model.pkl'))

    outcome_num = int(model.predict(X_pred)[0])
    if outcome_num == 0:
        outcome = "Oligodendroglioma"
        probability = round(float(model.predict_proba(X_pred)[0][0]), 4)
    else:
        outcome = "Astrocytoma"
        probability = round(float(model.predict_proba(X_pred)[0][1]), 4)

    return {
                "Outcome": outcome,
                "Probability": probability
    }

@app.get("/")
def root():
    return {'greeting': 'Hello'}

######### Run prediction of several samples ####################################
@app.post("/predict_several_samples")
def predict_several_samples(file: UploadFile = File(...)):
    contents = file.file.read()
    print(type(contents))
    decoded_str = contents.decode('utf-8')
    print(type(decoded_str))
    print(decoded_str)

    rows = decoded_str.split('\n')

    # Split each row into columns
    data = [row.split(',') for row in rows]

    # Convert the data into a DataFrame
    df_upload = pd.DataFrame(data[1:], columns=data[0])
    print(df_upload.head(3))

    df = df_upload.drop(columns="Identifier", axis = 1)

    # Preprocess
    X_pred = preprocess_input(df)

    # Predict data
    current_path = os.path.dirname(__file__)
    model =load(os.path.join(current_path,'..','models', 'log_reg_model.pkl'))
    outcome = pd.DataFrame(model.predict(X_pred), columns=["Outcome"], dtype = int)
    prob = pd.DataFrame(model.predict_proba(X_pred), columns=["Probability_0", "Probability_1"], dtype = float)

    # Merge results into dataframe
    result = pd.merge(prob,outcome, left_index=True, right_index=True)
    # Make the dataframe easily interpretable for the user
    result["Prediction"] = result["Outcome"].apply(lambda x: str("Oligodendroglioma") if x == 0 else str("Astrocytoma"))
    result["Probability"] = np.where(result["Outcome"] == 0,
                                    result["Probability_0"],
                                    result["Probability_1"])
    result_df = result[["Prediction", "Probability"]]
    result_df = df_upload[["Identifier"]].merge(result_df, left_index=True, right_index=True) # add idenfitier column from uploaded dataframe back to output

    # transform dataframe into list so it can be returned from api
    prediction = {k: v.tolist() for k, v in result_df.iterrows()}

    return prediction
