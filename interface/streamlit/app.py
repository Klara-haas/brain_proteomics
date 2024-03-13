import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px
import os


st.set_page_config(page_title="Proteomics Data for Earlier Brain Cancer Detection", page_icon="🧬", layout = "wide")

st.title('Predicting Brain Cancer with Proteomics')

st.write("")

path = os.path.dirname(__file__)
image_path = os.path.join(path,"images","eyewire.jpg")
image = Image.open(image_path)
st.image(image)

st.write("")

st.markdown("- <span style='font-size: 20px'> Proteomics is the study of proteins, and different cells will have different types and amounts of proteins. Like the genetic code in genomics, proteins can be used to try and find the 'fingerprints' of certain cells.</span>", unsafe_allow_html=True)
st.markdown("- <span style='font-size: 20px'> In this way we can classify different types of brain cancers, which can be otherwise difficult to determine. This would make it easier for doctors to try certain therapies and treatments sooner, and to improve patient success rates.</span>", unsafe_allow_html=True)
st.markdown("- <span style='font-size: 20px'> From a dataset of samples from two types of cancerous brain cells, we have developed a model to predict the cell type according to the presence of certain proteins.</span>", unsafe_allow_html=True)
