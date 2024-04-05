# Using Machine Learning and Proteomics Data for earlier Brain Cancer detection

## Motivation
Brain cancer affects thousands of individuals worldwide each year. To improve patient outcomes, it is important to determine the type, grade and size of the tumor in the early stage. Conventional diagnosis methods such as MRI and CT can visualize brain abnormalities, however these methods are time-consuming and expensive and can't distinguish between the two main types of brain cancer (Oligodendroglioma and Astrocytoma).
By using a proteomics approach, molecular patterns can be investigated to distinguish between the two tumor types in earlier brain cancer states. Investigation of proteomics data of ~ 300 patients suffering from oligodendroglioma and astrocytoma revealed 17 proteins that differ significantly in expression patterns between the two cancer types. Based on the proteomics data of these 17 proteins, we used a machine learning approach to predict the patients' cancer types.

## Code Style
standard python

## Tech/Framework used
We developed a machine learning framework including cleaning and preprocessing the raw-data, extracting relevant proteins for the model and using logistic regression to classify the patients outcome: Oligodendroglioma and Astrocytoma. 
Cleaning and preprocessing the raw data includes scaling the relevant proteins, which showed statistically different expression patterns between the two tumor types. As data points were limited, we created synthetic data using a SMOTE approach for the training set. We then performed a grid search for logistic regression parameters to minimize the recall of cancer type Astroglioma. The trained model was cross-validated with an evaluation set and tested afterwards with a test set to prevent data leakage. 

## How to Use?
The model’s prediction for the cancer type can be accessed through our user-friendly API and website: https://brainproteomicspredict.streamlit.app/Introduction. Here the user can upload a CSV-file containing the proteomics data of patients. The output shows the predicted cancer type and the probability. 

By running the ‘fast_api.py’ app locally, the user can also use the saved model to predict the cancer type of the input locally on the machine. 

## Credits
The dataset was provided on kaggle by Laila Qadir Musib: https://www.kaggle.com/datasets/leilahasan/brain-cancer-clinical-and-proteomic-data/data 
Model, API and Website was developed by Giorgos Valsamakis, Jana Schulz, Rebecca Pedinoff and Klara Haas 
