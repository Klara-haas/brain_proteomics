## Model & Prediction

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Model & Prediction", page_icon="📈")

#st.sidebar.header("Model & Prediction")

st.title("Model & Prediction")

st.header("Let's predict based on proteomics data!")

#st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache_data
data=None
data = st.file_uploader("Upload your data", type="csv")
df_upload=None

if data is not None:
    df_upload = pd.read_csv(data)
    st.success('File upload successful')
    st.markdown('**Uploaded data preview**')
    st.write(df_upload.head())
    #data.seek(0)
    st.markdown('**. . .**')


if df_upload is not None:
    fig = px.histogram(df_upload, x="years_to_birth",  color="gender", marginal="box",
                   hover_data=df_upload.columns)
    st.plotly_chart(fig)
else:
    st.write('Nothing uploaded yet')



# ######### Run prediction for 1 sample ##########################################
# #@app.post("/predict_uploaded_file")
# def predict_uploaded_file(file: UploadFile = File(...)):
#     contents = file.file.read()
#     print(type(contents))
#     decoded_str = contents.decode('utf-8')
#     print(type(decoded_str))
#     print(decoded_str)

#     rows = decoded_str.split('\n')

#     # Split each row into columns
#     data = [row.split(',') for row in rows]

#     # Convert the data into a DataFrame
#     df = pd.DataFrame(data[1:], columns=data[0])
#     print(df.head(3))

#     df = df.drop(columns="Identifier", axis = 1)


#     # Preprocess
#     X_pred = preprocess_input(df)

#     # Predict data
#     model = load_model(path ='/',
#                         file = 'sgd_model.pkl')

#     outcome_num = int(model.predict(X_pred)[0])
#     if outcome_num == 0:
#         outcome = "Oligodendroglioma"
#         probability = round(float(model.predict_proba(X_pred)[0][0]), 4)
#     else:
#         outcome = "Astrocytoma"
#         probability = round(float(model.predict_proba(X_pred)[0][1]), 4)

#     return {
#                 "Outcome": outcome,
#                 "Probability": probability
#     }




############ RUN PREDICTION FOR SEVERAL SAMPLES #####################

st.markdown('''
### Predict the type of cancer for all your uploaded samples:
''')
if st.button('Run prediction for all samples'):
    data.seek(0)

    # print is visible in the server output, not in the page

    #base_url = 'http://127.0.0.1:8000'
    base_url = 'https://brainproteomics-hnkdsog4wq-ew.a.run.app'
    endpoint = 'predict_several_samples'
    brainproteomics_api_url = f'{base_url}/{endpoint}'
    response = requests.post(brainproteomics_api_url, files={"file": data})

    prediction = response.json()
    st.write('Prediction was successful 🎉')

    #st.write(prediction)

    result_df = pd.DataFrame(prediction).transpose()
    result_df = result_df.rename(columns={0: "Identifier",
                                          1: "Prediction",
                                          2: "Probability"})

    st.write(result_df)

else:
    st.write('I was not clicked 😞')
