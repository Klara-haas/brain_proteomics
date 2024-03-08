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


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        return JSONResponse(status_code=400, content={"message": "This endpoint accepts only CSV files."})

    try:
        # Read the content of the file
        contents = await file.read()
        # Decode the contents
        decoded_contents = contents.decode("utf-8")
        # Use StringIO to convert the string data into a file-like object so it can be read by pandas
        data = StringIO(decoded_contents)
        # Use pandas to read the CSV data
        df = pd.read_csv(data)

        # Access the value of the first row in the first column
        first_value = df.iloc[0, 0]

        # Return the value
        return {"first_value": float(first_value)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    #content_type = file.content_type
    #try:
    contents = await file.read()

    # Handling CSV files
    #if content_type == "text/csv":
    #decoded_contents = contents.decode("utf-8")
    #data = StringIO(decoded_contents)
    #df = pd.read_csv(data)
        # Handling JSON files
        # elif content_type == "application/json":
    decoded_contents = contents.decode("utf-8")
    df_json = json.loads(decoded_contents)
    df = pd.DataFrame(df_json)
        # else:
        #     return JSONResponse(status_code=400, content={"message": "Unsupported file type. Please upload CSV or JSON."})

        # Access the value of the first row in the first column
    first_value = df.iloc[0, 0]

    # Return the value
    return {"first_value": float(first_value)}
    #except Exception as e:
    #return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/uploaded_file")
def upload_file(file: UploadFile = File(...)):
    contents = file.file.read() # Reading content of 'myfile' in bytes
    # print(contents)
    decoded_str = contents.decode('utf-8') # Decoding contents into str type
    #decoded_str = StringIO(contents.decode('utf-8')) # Alternative using StringIO
    df_json = json.loads(decoded_str) # Reading string and converting to json (dictionary)
    df = pd.DataFrame(df_json) # Reading dictionary and converting into dataframe
    results = {
        'mean1': float(df[1])
        }
    return results


@app.post("/predict_uploaded_file_test")
def predict_uploaded_file_test(file: UploadFile = File(...)):
    contents = file.file.read()
    print(type(contents))
    decoded_str = contents.decode('utf-8')
    print(type(decoded_str))
    print(decoded_str)
    print("test")
    #data_byte = data.to_json().encode()
    #df_json = json.loads(decoded_str)
    #print(df_json)
    #df = pd.DataFrame(df_json)
    #df = pd.read_csv(StringIO(decoded_str))

    rows = decoded_str.split('\n')

    # Split each row into columns
    data = [row.split(',') for row in rows]

    # Convert the data into a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])


    results = {
        'mean1': float(df["Identifier"][0])
        }
    return results

@app.post("/predict_uploaded_file_funny")
def predict_uploaded_file(file: UploadFile = File(...)):
    contents = file.file.read() # Reading content of 'myfile' in bytes
    #print(contents)
    decoded_str = contents.decode('utf-8') # Decoding contents into str type
    # decoded_str = StringIO(contents.decode('utf-8')) # Alternative using StringIO
    df_json = json.loads(decoded_str) # Reading string and converting to json (dictionary)
    df = pd.DataFrame(df_json) # Reading dictionary and converting into dataframe
    results = {
        "value": float(df["Identifier"][0])
        }
    return results


@app.post("/predict_uploaded_file")
def predict_uploaded_file(file: UploadFile = File(...)):
    contents = file.file.read() # Reading content of 'myfile' in bytes
    #print(contents)
    decoded_str = contents.decode('utf-8') # Decoding contents into str type
    # decoded_str = StringIO(contents.decode('utf-8')) # Alternative using StringIO
    df_json = json.loads(decoded_str) # Reading string and converting to json (dictionary)
    df = pd.DataFrame(df_json) # Reading dictionary and converting into dataframe
    # results = {
    #     "value": float(df["Identifier"][0])
    #     }
    # return results

    data = df.drop(columns="Identifier")


    # Preprocess
    X_pred = preprocess_input(data)


    # Predict data
    model = load_model(path ='/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_scalers',
                        file = 'svc_model.pkl')

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
