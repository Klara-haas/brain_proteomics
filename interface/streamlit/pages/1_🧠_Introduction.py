## introduction.py
import streamlit as st
from PIL import Image
import os

## 1 Introduction.py

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Introduction", page_icon="🧠", layout = "wide")


st.title('Introduction')

st.write("")
st.write("")

col1, col2 = st.columns([2, 2])

with col1:
    st.header("Glial cells and Gliomas")

    st.write("")

    intro_items1 = ["Glial cells: support neurons in the brain",
                "Gliomas: glial tumors",
                "Most common type of malignant brain tumors",
                "Classification according to histological criteria into:"
                ]
    st.markdown("\n".join([f"- <span style='font-size: 22px'>{item}</span>" for item in intro_items1]), unsafe_allow_html=True)

with col1:
    cola, colb = st.columns([0.2, 2])
    with colb:
        intro_items12 = [ "Oligodendroglioma",
                    "Oligoastrocytoma",
                    "Astrocytoma",
                    "Glioblastoma"
                        ]
        st.markdown("\n".join([f"- <span style='font-size: 22px'>{item}</span>" for item in intro_items12]), unsafe_allow_html=True)


with col2:
    path = os.path.dirname(__file__)
    image1_path = os.path.join(path,"..", "images","750px-Blausen_0870_TypesofNeuroglia_marked_red.png")
    #print("findme2", image1_path)
    image1 = Image.open(image1_path)
    st.image(image1, use_column_width=True)


st.write("")
st.write("")
st.write("")
st.write("")

col3, col4 = st.columns([2, 2], gap = "large")

with col3:
    image2_path = os.path.join(path,"..", "images","41374_2003_Article_BF3780627_Fig1_HTML.jpg")

    image2 = Image.open(image2_path)
    st.image(image2, caption='Histopathologic Classification of Gliomas', use_column_width=True)

with col4:
    st.header("The Problem")

    st.write("")

    intro_items1 = ["Difficult classification",
                    "Requires histological examination / biopsy",
                    "High variability between different cancer types"
                ]
    st.markdown("\n".join([f"- <span style='font-size: 22px'>{item}</span>" for item in intro_items1]), unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("")

col5, col6 = st.columns([2, 2], gap = "large")

with col6:
    image2_path = os.path.join(path,"..", "images","brain_proteomics_Wingo_et_al_2019.png")

    image2 = Image.open(image2_path)
    st.image(image2, use_column_width=True)

with col5:
    st.header("The Solution")

    st.write("")

    intro_items1 = ["Proteins as key players in early detection!",
                    "Faster diagnosis",
                    "More targeted treatment",
                    "Better patient outcomes"
                ]
    st.markdown("\n".join([f"- <span style='font-size: 22px'>{item}</span>" for item in intro_items1]), unsafe_allow_html=True)
