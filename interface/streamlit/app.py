import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px


st.set_page_config(page_title="Proteomics Data for Earlier Brain Cancer Detection", page_icon="🧬", layout = "wide")

st.title('Using Machine Learning with Proteomics Data for Earlier Brain Cancer Detection')

st.write("""
         Proteomics is the study of proteins, and different cells will have different types and amounts of proteins.
         Like the genetic code in genomics, proteins can be used to try and find the "fingerprints" of certain cells.

         In this way we can classify different types of brain cancers, which can be otherwise difficult to determine.
         This would make it easier for doctors to try certain therapies and treatments sooner, and to improve patient success rates.

         From a dataset of samples from two types of cancerous brain cells, we have developed a model to predict the cell type according to the presence of certain proteins.
         """)
