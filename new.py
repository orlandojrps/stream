import streamlit as st
import numpy as np
import joblib
from io import BytesIO
import requests
mLink = 'https://github.com/orlandojrps/stream/blob/main/model.pkl?raw=true'
mfile = BytesIO(requests.get(mLink).content)

model = joblib.load(mfile)

# Load the pre-trained linear regression model
#lr_model = joblib.load(model)


# Define the function to make a prediction
def predict_chd_risk(features):
    prediction = model.predict(features)
    return prediction[0]

# Define the Streamlit app
def app():
    st.title('CHD Risk Prediction App')
    st.write('Enter the following information to predict your CHD risk:')
    age = st.slider('Age', 25, 80, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 500, 240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Probable/Definite Left Ventricular Hypertrophy'])
    thalach = st.slider('Maximum Heart Rate Achieved', 50, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 2.0, 0.1)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'])
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    if sex == 'Male':
        sex_val = 1
    else:
        sex_val = 0
    if cp == 'Typical Angina':
        cp_val = 0
    elif cp == 'Atypical Angina':
        cp_val = 1
    elif cp == 'Non-anginal Pain':
        cp_val = 2
    else:
        cp_val = 3
    if fbs == 'False':
        fbs_val = 0
    else:
        fbs_val = 1
    if restecg == 'Normal':
        restecg_val = 0
    elif restecg == 'ST-T Abnormality':
        restecg_val = 1
    else:
        restecg_val = 2
    if exang == 'No':
        exang_val = 0
    else:
        exang_val = 1
    features = np.array([age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = predict_chd_risk(features)
    st.write('Your predicted CHD risk is:', prediction)
