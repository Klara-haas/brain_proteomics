import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(layout="wide") # sets layout to wide

# Input widgets in the sidebar
with st.sidebar:
	""" ### Navigation"""
	# input widget 2
	...

# Load data based on the inputs from the sidebar widgets

# Main body contents: Output Widgets
st.title("Proteomomic-based Brain Cancer Prediction")
st.header("Oligodendroglioma or Astrocytoma?")
st.markdown('''
**Let's predict based on proteomics data!** \n
**Upload your data here:**
''')

#st.set_option('deprecation.showfileUploaderEncoding', False)

data = st.file_uploader("Choose a CSV file", type="csv")

if st.button('Show my uploaded data'):
    # print is visible in the server output, not in the page
    df = pd.read_csv(data)
    st.write(df.head(3))

data.seek(0) # to reload data into the buffer


# st.markdown('''
# **Test**:
# This is a test to make sure that the data can be loaded and passed to the api.
# If it works it will return the value of first column, first row in the df.
# ''')
# # try with endpoint predict_uploaded_file_test
# # include string concatenation to avoid errors
# # this works
# base_url = 'http://127.0.0.1:8000'
# endpoint = 'predict_uploaded_file_test'
# brainproteomics_api_url = f'{base_url}/{endpoint}'
# response = requests.post(brainproteomics_api_url, files={"file": data})
# prediction = response.json()
# st.write(prediction)


############# RUN PREDICTION FOR 1 SAMPLE ######################################
data.seek(0)
st.markdown('''
### To predict the type of cancer based on your uploaded data click predict:
''')
# try with endpoint predict_uploaded_file_test
# include string concatenation to avoid errors
# this works
base_url = 'http://127.0.0.1:8000'
endpoint = 'predict_uploaded_file'
brainproteomics_api_url = f'{base_url}/{endpoint}'
response = requests.post(brainproteomics_api_url, files={"file": data})

prediction = response.json()


if st.button('Predict!'):
    # print is visible in the server output, not in the page

    st.write('Prediction was successful 🎉')
    st.subheader(f'Cancer type: {prediction["Outcome"]}')
    st.subheader(f'Probability: {prediction["Probability"]}')
else:
    st.write('I was not clicked 😞')


############# RUN PREDICTION FOR SEVERAL SAMPLE ################################
data.seek(0)
st.markdown('''
### To predict the type of cancer for all your uploaded samples:
''')
if st.button('Run prediction for all samples'):
    # print is visible in the server output, not in the page

    base_url = 'http://127.0.0.1:8000'
    endpoint = 'predict_several_samples'
    brainproteomics_api_url = f'{base_url}/{endpoint}'
    response = requests.post(brainproteomics_api_url, files={"file": data})

    prediction = response.json()
    st.write('Prediction was successful 🎉')
    st.write(prediction)

else:
    st.write('I was not clicked 😞')
