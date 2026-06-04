import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Customer Churn Predictor')
st.write('Enter customer details below to predict if they will churn')

# Input fields
tenure = st.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.slider('Monthly Charges ($)', 18, 119, 64)
total_charges = tenure * monthly_charges

senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Has Partner', ['No', 'Yes'])
dependents = st.selectbox('Has Dependents', ['No', 'Yes'])
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
payment_method = st.selectbox('Payment Method', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])

# Build input dataframe matching training columns
input_dict = {
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'gender_Male': [0],
    'Partner_Yes': [1 if partner == 'Yes' else 0],
    'Dependents_Yes': [1 if dependents == 'Yes' else 0],
    'PhoneService_Yes': [1 if phone_service == 'Yes' else 0],
    'MultipleLines_No phone service': [1 if multiple_lines == 'No phone service' else 0],
    'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0],
    'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
    'InternetService_No': [1 if internet_service == 'No' else 0],
    'OnlineSecurity_No internet service': [1 if online_security == 'No internet service' else 0],
    'OnlineSecurity_Yes': [1 if online_security == 'Yes' else 0],
    'OnlineBackup_No internet service': [1 if online_backup == 'No internet service' else 0],
    'OnlineBackup_Yes': [1 if online_backup == 'Yes' else 0],
    'DeviceProtection_No internet service': [1 if device_protection == 'No internet service' else 0],
    'DeviceProtection_Yes': [1 if device_protection == 'Yes' else 0],
    'TechSupport_No internet service': [1 if tech_support == 'No internet service' else 0],
    'TechSupport_Yes': [1 if tech_support == 'Yes' else 0],
    'StreamingTV_No internet service': [1 if streaming_tv == 'No internet service' else 0],
    'StreamingTV_Yes': [1 if streaming_tv == 'Yes' else 0],
    'StreamingMovies_No internet service': [1 if streaming_movies == 'No internet service' else 0],
    'StreamingMovies_Yes': [1 if streaming_movies == 'Yes' else 0],
    'Contract_One year': [1 if contract == 'One year' else 0],
    'Contract_Two year': [1 if contract == 'Two year' else 0],
    'PaperlessBilling_Yes': [1 if paperless_billing == 'Yes' else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
}

input_df = pd.DataFrame(input_dict)

# Predict button
if st.button('Predict Churn'):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f'⚠️ This customer is likely to churn! ({probability:.0%} probability)')
    else:
        st.success(f'✅ This customer is likely to stay! ({1-probability:.0%} probability)')