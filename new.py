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
   
    features = np.array([[age,	0,	0,	100,	248,	0,	0,	122,	0,	1.0,	1,	0,	2]])
    prediction = predict_chd_risk(features)
    st.write('Your predicted CHD risk is:', prediction)
    st.write('array',features)
    

if __name__ == "__main__":
    app()
    
