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
#st.set_option('deprecation.showfileUploaderEncoding', False)

data = st.file_uploader("Choose a CSV file", type="csv")
#breakpoint()
if data:
    df = pd.read_csv(data)
    st.write(df.head(3))

st.write(df["Identifier"])

data.seek(0) # to reload data into the buffer

#data_byte = df.to_json().encode()
#st.write(type(data_byte))
#data_byte = data.encode()

# try with /predict_uploaded_file_test
#brainproteomics_api_url = 'http://127.0.0.1:8000/predict_uploaded_file_test'
#response = requests.post(brainproteomics_api_url, files={"file": data})

#prediction = response.json()

#st.write(prediction)

st.markdown('''
**Test**:
This is a test to make sure that the data can be loaded and passed to the api.
If it works it will return the value of first column, first row in the df.
''')
# try with endpoint predict_uploaded_file_test
# include string concatenation to avoid errors
# this works
base_url = 'http://127.0.0.1:8000'
endpoint = 'predict_uploaded_file_test'
brainproteomics_api_url = f'{base_url}/{endpoint}'
response = requests.post(brainproteomics_api_url, files={"file": data})

prediction = response.json()

st.write(prediction)


data.seek(0)
st.markdown('''
### Make Prediction
Run prediction with SVC model
''')
# try with endpoint predict_uploaded_file_test
# include string concatenation to avoid errors
# this works
base_url = 'http://127.0.0.1:8000'
endpoint = 'predict_uploaded_file'
brainproteomics_api_url = f'{base_url}/{endpoint}'
response = requests.post(brainproteomics_api_url, files={"file": data})

prediction = response.json()

st.write(prediction)



# here is some problem currently
#brainproteomics_api_url = 'http://127.0.0.1:8000/predict_uploaded_file'
#response = requests.post(brainproteomics_api_url, files={"file": data_byte})

#prediction = response.json()

#st.write(prediction)





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
