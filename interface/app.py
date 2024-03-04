# Hello world!

import streamlit as st
import requests

'''
# BrainProteomics front
'''

st.markdown('''
Here is the frontend of the app
This front queries the Brain Proteomics [API](https://.../predict?param0=0&param1=1)
''')


'''
with st.form(key='params_for_api'):

    param 0 = st.function('param0', value=0)
    param1 = st.function('param1', value=1)

    st.form_submit_button('Make prediction')

params = dict(
    param0=param0,
    param1=param1)
'''

brainproteomics_api_url = 'https://.../predict'
#response = requests.get(brainproteomics_api_url, params=params)

#prediction = response.json()

#pred = prediction['my prediction']

st.header(f'My Prediction: 100')            # $(pred)')

''''''
df
histogram
heatmap
''''''
