## Model & Prediction

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px
import os

st.set_page_config(page_title="Model & Prediction", page_icon="📈", layout = "wide")

#st.sidebar.header("Model & Prediction")

st.title("Model & Prediction")

st.header("Let's predict with machine learning and proteomics data!")

#st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache_data
st.markdown(
    """
    <style>
    .css-1v3fvcr {
        font-size: 22px; /* Adjust the font size as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

data=None
st.write('Upload your data')
data = st.file_uploader("", type="csv")
df_upload=None

#f"- <span style='font-size: 20px'>{text}</span>", unsafe_allow_html=True

if data is not None:
    df_upload = pd.read_csv(data)
    st.success('File upload successful')
    st.markdown('**Uploaded data preview**')
    st.write(df_upload.head())
    #data.seek(0)
    st.markdown('**. . .**')


if df_upload is not None:
    fig = px.histogram(df_upload, x="years_to_birth",  color="gender", marginal="box",
                   hover_data=df_upload.columns,
                   color_discrete_map={'female': '#81689D', 'male': '#FFD0EC'})
    fig.update_layout(
            xaxis_title="Years to birth",
            yaxis_title="Count",
            legend_title="Gender",
            xaxis = dict(title_font = dict(size = 20),
                         tickfont = dict(size = 20)),
            yaxis = dict(title_font = dict(size = 20),
                         tickfont = dict(size = 20)),
            legend=dict(
                font = dict(size = 18),
                title_font = dict(size = 20)
                )
            )

    st.plotly_chart(fig)
else:
    st.write('Nothing uploaded yet')




############# RUN PREDICTION FOR SEVERAL SAMPLE ################################

st.markdown('''
### Predict the type of cancer for all samples:
''')
if st.button('Run prediction'):
    data.seek(0)

    #base_url = 'http://127.0.0.1:8000' #local
    #base_url = "https://brainproteomics-hnkdsog4wq-ew.a.run.app" #dockerfile 12/03/2024
    base_url = "https://brainproteomicsnew-hnkdsog4wq-ew.a.run.app" #dockerfile 13/03/2024
    endpoint = 'predict_several_samples'
    brainproteomics_api_url = f'{base_url}/{endpoint}'
    response = requests.post(brainproteomics_api_url, files={"file": data})

    prediction = response.json()
    st.write('Prediction was successful 🎉')

    result_df = pd.DataFrame(prediction).transpose()
    result_df = result_df.rename(columns={0: "Identifier",
                                          1: "Prediction",
                                          2: "Probability"})

    # Show result dataframe with prediction

    ca, cb = st.columns([2.2,1.8], gap = "large")
    with ca:
        st.subheader("Results of prediction")
        st.write("")
        st.dataframe(result_df, use_container_width=True)

    # Pie chart
    with cb:
        cb.subheader("Proportion of predicted cancer types")
        st.write("")

        tmp = pd.DataFrame(result_df[["Prediction"]].value_counts())
        fig = px.pie(tmp,
                     values=  result_df["Prediction"].value_counts().values,
                     names = result_df["Prediction"].sort_values().unique(),
                     color_discrete_map={'Oligodendroglioma': '#265073', 'Astrocytoma': '#2D9596'}
                     #color=result_df["Prediction"].unique().sort()
                     )

        fig.update_traces(marker=dict(colors=['#265073', '#2D9596']))

        fig.update_layout(
            height = 450,
            width = 450,
            #title="Plot Title",
            xaxis_title=None,
            xaxis = dict(title_font = dict(size = 20),
                         tickfont = dict(size = 20)),
            yaxis = dict(title_font = dict(size = 20),
                         tickfont = dict(size = 20)),
            font=dict(size=18),
            legend_title=None,
            legend=dict(
                font = dict(size = 20),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
                )
            )

        st.plotly_chart(fig,height=500, width = 450)


    st.write("")
    st.write("")
    st.write("")
    st.write("")

    #c2, c3 = st.columns([1.5, 2.5], gap = "medium")

    # Boxplot of probabilities
    # with c2:
    # st.subheader("Probability of predicted cancer types")
    # st.write("")

    # fig = px.box(result_df, x = "Prediction", y='Probability', points = "outliers",
    #                 color='Prediction',
    #                 color_discrete_map={'Oligodendroglioma': '#2D9596', 'Astrocytoma': '#265073'}
    #                 )
    #             #category_orders={'outcome': ['Oligodendroglioma', 'Astrocytoma']}
    #             #)

    # fig.update_yaxes(tickfont=dict(size=20))
    # fig.update_xaxes(tickfont=dict(size=20))
    # #fig.update_traces(marker=dict(colors=['red', 'blue']))
    # fig.update_layout(
    #     height = 450,
    #     width = 450,
    #     #title="Plot Title",
    #     xaxis_title=None,
    #     yaxis_title="Probability",
    #     yaxis = dict(title_font = dict(size = 20)),
    #     yaxis_range=[0,1],
    #     legend_title=None,
    #     font=dict(size=18),
    #     legend=dict(
    #         font = dict(size = 20),
    #         orientation="h",
    #         yanchor="bottom",
    #         y=1.02,
    #         xanchor="center",
    #         x=0.5
    #         )
    #     )
    # st.plotly_chart(fig,height=500, width = 500)

    ###### Last Page ###############################################################
    st.write("")
    st.write("")
    st.write("")

    st.markdown("# We can predict a patient's cancer type without invasive methods!")
    path = os.path.dirname(__file__)
    image2_path = os.path.join(path,"..", "images","Final.jpg")

    image2 = Image.open(image2_path)
    st.image(image2, use_column_width=True)

else:
    pass






############### Unused code ####################################################

# HISTOGRAM
    # # Histogram
    # with c2:
    #     c2.subheader("Distribution of prediction probabilities")
    #     st.write("")
    #     #df = px.data.result_df
    #     #fig = px.histogram(df, x="total_bill")
    #     fig = px.histogram(result_df, x="Probability", nbins = 20, color = "Prediction",
    #                 marginal="box", # Add marginal rug plots on the axes
    #                 color_discrete_map={'Oligodendroglioma': '#2D9596', 'Astrocytoma': '#265073'})

    #     fig.update_yaxes(tickfont=dict(size=20))
    #     fig.update_xaxes(tickfont=dict(size=20))
    #     fig.update_layout(
    #         height=450,
    #         width=450,
    #         xaxis_title= "Probability",
    #         yaxis_title="Count",
    #         yaxis=dict(title_font=dict(size=20)),
    #         xaxis=dict(title_font=dict(size=20)),
    #         xaxis_range=[0, 1],
    #         legend_title=None,
    #         font=dict(size=18),
    #         legend=dict(
    #             font=dict(size=20),
    #             orientation="h",
    #             yanchor="bottom",
    #             y=1.02,
    #             xanchor="center",
    #             x=0.5
    #         )
    #     )
