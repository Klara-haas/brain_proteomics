from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from brain_proteomics.data_preproc.preprocess import preprocess_data
### import preprocessor X_train,y_train, X_val, y_val


# train the model
def train_model(X_train, y_train):
    log_reg_model = LogisticRegression(max_iter=1000)

    log_reg_model.fit(X_train, y_train)
    return log_reg_model

# Evaluate the model
def val_model(log_reg_model, X_val, y_val):
    y_pred_log = log_reg_model.predict(X_val)
    y_val= y_val
    accuracy = accuracy_score(y_val, y_pred_log)
    print("Accuracy:", accuracy)

    # Classification report
    classification= classification_report(y_val, y_pred_log)
    print("\nClassification Report:", classification)

    return classification

# save the model

def save_model(log_reg_model):
    current_path = os.path.dirname(__file__)
    joblib.dump(log_reg_model, (os.path.join(current_path, 'log_reg_model.pkl')))


#X_train, y_train, X_val, y_val, X_test, y_test, preproc_base = preprocess_data()
#log_reg_model = train_model(X_train, y_train)
#classification_report = val_model(log_reg_model, X_val,y_val)
