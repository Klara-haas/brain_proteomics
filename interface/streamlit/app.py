import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px
import os


st.set_page_config(page_title="Proteomics Data for Earlier Brain Cancer Detection", page_icon="🧬", layout = "wide")


#st.title('Predicting Brain Cancer with Proteomics')

#st.write("")

path = os.path.dirname(__file__)
image_path = os.path.join(path,"images","eyewire_title.jpg")
#image_path = os.path.join(path,"images","Using Machine Learning and Proteomics Data for earlier Brain Cancer detection.jpg")
image = Image.open(image_path)
st.image(image)

st.markdown("<span style='font-size: 20px'> Brain cancer affects thousands of individuals worldwide each year. To improve patient outcomes, it is important to determine the type, grade and size of the tumor in the early stage. Conventional diagnosis methods such as MRI and CT can visualize brain abnormalities, however these methods are time-consuming and expensive and can't distinguish between the two main types of brain cancer (Oligodendroglioma and Astrocytoma).  </span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 20px'> By using a proteomics approach, molecular patterns can be investigated to distinguish between the two tumor types in earlier brain cancer states. Investigation of proteomics data of ~ 300 patients suffering from oligodendroglioma and astrocytoma revealed 17 proteins that differ significantly in expression patterns between the two cancer types. Based on the proteomics data of these 17 proteins, we used a machine learning approach to predict the patients' cancer types. </span>", unsafe_allow_html=True)


#st.markdown("- <span style='font-size: 20px'> Proteomics is the study of proteins, and different cells will have different types and amounts of proteins. Like the genetic code in genomics, proteins can be used to try and find the 'fingerprints' of certain cells.</span>", unsafe_allow_html=True)
#st.markdown("- <span style='font-size: 20px'> In this way we can classify different types of brain cancers, which can be otherwise difficult to determine. This would make it easier for doctors to try certain therapies and treatments sooner, and to improve patient success rates.</span>", unsafe_allow_html=True)
#st.markdown("- <span style='font-size: 20px'> From a dataset of samples from two types of cancerous brain cells, we have developed a model to predict the cell type according to the presence of certain proteins.</span>", unsafe_allow_html=True)
