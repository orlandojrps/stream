import streamlit as st
import numpy as np
import joblib
from io import BytesIO
import requests
import warnings
warnings.filterwarnings("ignore")

mLink = 'https://github.com/orlandojrps/stream/blob/main/model.pkl?raw=true'
mfile = BytesIO(requests.get(mLink).content)

model = joblib.load(mfile)

# Load the pre-trained linear regression model
#lr_model = joblib.load(model)


# Define the function to make a prediction
def predict_chd_risk(features):
    prediction = model.predict(features)
    return prediction[0]

st.title('CHD Risk Prediction App')
# Define the Streamlit app
def app():
    st.title('CHD Risk Prediction App')
    st.write('Enter the following information to predict your CHD risk:')
    age = st.slider('Age', 25, 80, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    if sex == 'Male':
        sex_val = 1
    else:
        sex_val = 0
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    if cp == 'Typical Angina':
        cp_val = 0
    elif cp == 'Atypical Angina':
        cp_val = 1
    elif cp == 'Non-anginal Pain':
        cp_val = 2
    else:
        cp_val = 3
   
    features = np.array([[age,	sex_val,	cp_val,	100,	248,	0,	0,	122,	0,	1.0,	1,	0,	2]])
    prediction = predict_chd_risk(features)
    st.write('Your predicted CHD risk is:', prediction)
    st.write('array',features)
    

if __name__ == "__main__":
    app()
    
