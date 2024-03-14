## 2 About.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import random
import os


st.set_page_config(page_title="About the Dataset", page_icon="📊", layout = "wide")

st.title("Proteomic-based Brain Cancer Prediction")

# Read the data
path = os.path.dirname(__file__)
df_path = os.path.join(path,"..","brain-proteomics.csv")
#print("findme", df_path)
df = pd.read_csv(df_path)

gender_age_outcome = df[['years_to_birth', 'gender', 'outcome']]
gender_age_outcome.rename(columns={'years_to_birth': 'age'}, inplace=True)
outcome_mapping = {0: 'Oligodendroglioma', 1: 'Astrocytoma'}
gender_age_outcome['outcome'] = gender_age_outcome['outcome'].map(outcome_mapping)
outcome_percentages = gender_age_outcome['outcome'].value_counts(normalize=True) * 100


items = ["2 types of brain cancer",
        "305 samples",
        "176 proteins + age + gender (features)"
]

# Display the list as bulletpoints using Markdown
st.markdown("## About the dataset:")
st.markdown("\n".join([f"- <span style='font-size: 20px'>{item}</span>" for item in items]), unsafe_allow_html=True)



st.header("How many datapoints per cancer type?")
# Plotly figure 1
fig1 = px.pie(names=outcome_percentages.index, values=outcome_percentages.values,
              #title='Distribution of the two different outcomes',
              labels=outcome_percentages.index)

fig1.update_layout(
            height = 450,
            width = 450,
            xaxis_title=None,
            xaxis = dict(title_font = dict(size = 16),
                         tickfont = dict(size = 16)),
            yaxis = dict(title_font = dict(size = 16),
                         tickfont = dict(size = 16)),
            font=dict(size=14),
            legend_title=None,
            legend=dict(
                font = dict(size = 16),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
                )
            )

fig1.update_traces(marker=dict(colors=['#265073', '#2D9596']))

# Display figure 1
st.plotly_chart(fig1)



# Add text between figures
st.header("Age and gender distribution")

# Plotly figure 2
fig2 = px.histogram(gender_age_outcome, x="age", color="gender", marginal="box",
                    hover_data=gender_age_outcome.columns,
                    color_discrete_map={'female': '#81689D', 'male': '#FFD0EC'})

fig2.update_layout(
            xaxis_title="Years to birth",
            yaxis_title="Count",
            legend_title="Gender",
            xaxis = dict(title_font = dict(size = 16),
                         tickfont = dict(size = 16)),
            yaxis = dict(title_font = dict(size = 16),
                         tickfont = dict(size = 16)),
            legend=dict(
                font = dict(size = 14),
                title_font = dict(size = 16)
                )
            )

# Display figure 2
st.plotly_chart(fig2)


# Add text between figures
items2 = ["Statistical comparison between the two cancer types",
        "17 significantly different proteins",
        "Including age as a feature / excluding gender"
]

st.markdown("## To reduce analysis complexity:")
st.markdown("\n".join([f"- <span style='font-size: 20px'>{item}</span>" for item in items2]), unsafe_allow_html=True)


P17_list = ['Syk_p',
 'YAP_pS127_p',
 'AR_p',
 'ACC1_p',
 'YAP_p',
 'HER3_pY1289_p',
 'c-Kit_p',
 'ACC_pS79_p',
 'STAT3_pY705_p',
 'DJ-1_p',
 '53BP1_p',
 'p27_p',
 'PDK1_p',
 'S6_pS235_S236_p',
 'PRDX1_p',
 'Bax_p',
 'IRS1_p']

st.header("Examples from the 17 selected proteins")

# Select 5 random proteins from the P17_list
random_proteins = random.sample(P17_list, 5)

# Melt the dataframe to convert the selected columns into a single column
melted_df_random = df.melt(id_vars='outcome', value_vars=random_proteins, var_name='Proteins', value_name='Protein Levels')

# Replace numeric values in the 'outcome' column with their corresponding labels
melted_df_random['outcome'] = melted_df_random['outcome'].map(outcome_mapping)

# Plot the boxplot using Plotly
fig3 = px.box(melted_df_random, x='Proteins', y='Protein Levels', color='outcome',
             #title='Boxplot of 5 Randomly Selected Proteins',
             category_orders={'outcome': ['Oligodendroglioma', 'Astrocytoma']},
             color_discrete_map={'Oligodendroglioma': '#2D9596', 'Astrocytoma': '#265073'})

# Display figure 3
st.plotly_chart(fig3)



items3 = ["18 features (17 selected proteins + age)",
        "Production of synthetic data",
        "Logistic Regression"
]
st.markdown("## Final Analysis:")
st.markdown("\n".join([f"- <span style='font-size: 20px'>{item}</span>" for item in items3]), unsafe_allow_html=True)

st.divider()
st.write('📊 Original Dataset found at:')
st.markdown('[https://www.kaggle.com/datasets/leilahasan/brain-cancer-clinical-and-proteomic-data/data](https://www.kaggle.com/datasets/leilahasan/brain-cancer-clinical-and-proteomic-data/data)')
