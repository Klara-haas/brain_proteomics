# Hello world!

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

'''
# BrainProteomics front
'''

st.markdown('''
Here is the frontend of the app
This front queries the Brain Proteomics [API](https://.../predict?param0=0&param1=1)
''')


st.markdown('''
Oligodendroglioma or Astrocytoma? Let's predict based on proteomics data!
Upload your data here:
''')
st.set_option('deprecation.showfileUploaderEncoding', False)

df = st.file_uploader("Choose a CSV file", type="csv")

if df is not None:
    data = pd.read_csv(df)
    st.write(data.head(3))

data_byte = data.to_json().encode()

brainproteomics_api_url = 'http://127.0.0.1:8000/predict_uploaded_file'
response = requests.post(brainproteomics_api_url, files={"file": data_byte})

prediction = response.json()

st.write(prediction)





'''
with st.form(key='params_for_api'):

    param 0 = st.function('param0', value=0)
    param1 = st.function('param1', value=1)

    st.form_submit_button('Make prediction')

params = dict(
    param0=param0,
    param1=param1)
'''



########## Code to upload file using file path and file name ############################################
#'''
#1. Please add the following informations to get your prediction:
#- path to your file
#- file name
#'''

# path = st.text_input('path1', '/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/raw_data')
# file = st.text_input('file1', 'Glioma-clinic-TCGA-proteins-test-with-identifier-outcome1.csv')


# params = {'path': path,
#           'file': file}
# brainproteomics_api_url = 'http://127.0.0.1:8000/predict_one'
# response = requests.get(brainproteomics_api_url, params=params)

# prediction = response.json()

# st.write(prediction)
