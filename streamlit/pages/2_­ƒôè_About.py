## About.py
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import random


st.set_page_config(page_title="About the Dataset", page_icon="📊")

#st.sidebar.header("About the Dataset")

st.title('About the Dataset')


# Read the data
df = pd.read_csv('brain-proteomics.csv')

gender_age_outcome = df[['years_to_birth', 'gender', 'outcome']]
gender_age_outcome.rename(columns={'years_to_birth': 'age'}, inplace=True)
outcome_mapping = {0: 'Oligodendroglioma', 1: 'Astrocytoma'}
gender_age_outcome['outcome'] = gender_age_outcome['outcome'].map(outcome_mapping)
outcome_percentages = gender_age_outcome['outcome'].value_counts(normalize=True) * 100



items = ["2 types of brain cancer",
        "305 samples",
        "176 proteins (features)",
        "Age and gender (features)"
]

# Display the list as bulletpoints using Markdown
st.markdown("## About the dataset:")
st.markdown("\n".join([f"- {item}" for item in items]))


st.header("How many datapoints per cancer type?")
# Plotly figure 1
fig1 = px.pie(names=outcome_percentages.index, values=outcome_percentages.values,
              #title='Distribution of the two different outcomes',
              labels=outcome_percentages.index)

# Display figure 2
st.plotly_chart(fig1)

# Add text between figures
st.header("Age and gender distribution")

# Plotly figure 2
fig2 = px.histogram(gender_age_outcome, x="age", color="gender", marginal="box",
                    hover_data=gender_age_outcome.columns)

# Display figure 1
st.plotly_chart(fig2)


# Add text between figures
st.header("Afraid of the curse of dimentionality?")
st.write("Yes!")

items2 = ["Statistical analysis bewtween the two cancer types",
        "17 significantly expressed proteins!",
        "Age was a significant feature",
        "Gender was not statistically significant"
]
st.markdown("## Regarding dimetionality reduction:")
st.markdown("\n".join([f"- {item}" for item in items2]))


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

st.header("5 of the 17 selected proteins based on statistics")

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
             color_discrete_map={'Oligodendroglioma': 'blue', 'Astrocytoma': 'red'})

# Display figure 1
st.plotly_chart(fig3)

items3 = ["18 features (17 significantly expressed proteins + Age)",
        "Production of Synthetic data",
        "Logistic Regression"
]
st.markdown("## About the training dataset:")
st.markdown("\n".join([f"- {item}" for item in items3]))



st.write('Original Dataset found at:')
st.markdown('[https://www.kaggle.com/datasets/leilahasan/brain-cancer-clinical-and-proteomic-data/data](https://www.kaggle.com/datasets/leilahasan/brain-cancer-clinical-and-proteomic-data/data)')
