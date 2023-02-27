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
