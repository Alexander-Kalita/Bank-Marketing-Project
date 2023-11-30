
import json
import joblib
import requests
import pandas as pd
import xgboost as xgb
import streamlit as st

# Create Depositer app in Streamlit for new customer predictions
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_cv.json")

st.title('Depositer')
st.write("Please provide the customer information below to predict the probability of subscribing to a bank deposit.")

def user_input_features():
    age = st.slider('Age (in years)', 18, 130, 18)
    balance = st.slider('Balance (in EUR)', -10000000, 10000000, 0)
    day = st.slider('Last Contact Day of the Month', 1, 31, 1)
    duration = st.slider('Duration of Last Contact (in seconds)', 0, 10000, 0)
    campaign = st.slider('Number of Contacts Performed During This Campaign', 1, 1000, 1)
    pdays = st.slider('Number of Days Passed After the Client Was Last Contacted (-1 means not previously contacted)', -1, 1000, -1)
    previous = st.slider('Number of Contacts Performed Before This Campaign', 0, 1000, 0)

    job = ['Blue-Collar', 'Entrepreneur', 'Housemaid', 'Management', 'Retired', 'Self-employed',
           'Services', 'Student', 'Technician', 'Unemployed', 'Unknown']
    job_selected = st.selectbox("Job Type", job)
    job_category = {f'job_{category.lower()}': 1 if category == job_selected else 0 for category in job}
    
    education = ['Secondary', 'Tertiary', 'Unknown']
    education_selected = st.selectbox("Education Level", education)
    education_category = {f'education_{category.lower()}': 1 if category == education_selected else 0 for category in education}

    marital = ['Married', 'Single']
    marital_selected = st.selectbox("Marital Status", marital)
    marital_category = {f'marital_{category.lower()}': 1 if category == marital_selected else 0 for category in marital}

    contact =  ['Telephone', 'Unknown']
    contact_selected = st.selectbox("Contact Communication Type", contact)
    contact_category = {f'contact_{category.lower()}': 1 if category == contact_selected else 0 for category in contact}

    month = ['January', 'February', 'March', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']
    month_selected = st.selectbox('Contact Month', month)
    month_category = {f'month_{category.lower()}': 1 if category == month_selected else 0 for category in month}
    
    poutcome = ['Success', 'Other', 'Unknown']
    poutcome_selected = st.selectbox('Previous Campaign Outcome', poutcome)
    poutcome_category = {f'poutcome_{category.lower()}': 1 if category == poutcome_selected else 0 for category in poutcome}

    default_yes = st.checkbox('Credit Default (Yes)')
    housing_yes = st.checkbox('Housing Loan (Yes)')
    loan_yes = st.checkbox('Personal Loan (Yes)')
    
    user_features = {
        'age': age,
        'balance': balance,
        'day': day,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'job_blue-collar': job_category,
        'job_entrepreneur': job_category,
        'job_housemaid': job_category,
        'job_management': job_category,
        'job_retired': job_category,
        'job_self-employed': job_category,
        'job_services': job_category,
        'job_student': job_category,
        'job_technician': job_category,
        'job_unemployed': job_category,
        'job_unknown': job_category,
        'education_secondary': education_category,
        'education_tertiary': education_category,
        'education_unknown': education_category,
        'marital_married': marital_category,
        'marital_single': marital_category,
        'contact_telephone': contact_category,
        'contact_unknown': contact_category,
        'month_aug': month_category,
        'month_dec': month_category,
        'month_feb': month_category,
        'month_jan': month_category,
        'month_jul': month_category,
        'month_jun': month_category,
        'month_mar': month_category,
        'month_may': month_category,
        'month_nov': month_category,
        'month_oct': month_category,
        'month_sep': month_category,
        'poutcome_other': poutcome_category,
        'poutcome_success': poutcome_category,
        'poutcome_unknown': poutcome_category,
        'default_yes': default_yes,
        'housing_yes': housing_yes,
        'loan_yes': loan_yes
    }
    return user_features

user_features = user_input_features()
if st.button('Predict'):
    flattened_features = {}
    for key, value in user_features.items():
        if isinstance(value, dict):
            flattened_features.update(value)
        else:
            flattened_features[key] = value
    url = "http://172.19.0.2:8081/predict"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json={'features': flattened_features}, headers=headers)
    if response and response.status_code == 200:
        prediction = response.json().get('prediction', None)
        st.success(f'Deposit Subscription Probability: {prediction:.2f}')
    elif response:
        st.error("Failed to get prediction from the server.")
