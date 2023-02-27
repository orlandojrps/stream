import streamlit as st
import numpy as np
import joblib
from io import BytesIO
import requests
import matplotlib.pyplot as plt

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
        # Set the background image
    st.set_page_config(page_title='CHD Risk Prediction App', page_icon=':heart:', layout='wide', 
                       initial_sidebar_state='auto', 
                       page_bg_image='https://previews.123rf.com/images/nexusplexus/nexusplexus1306/nexusplexus130601789/20326618-illustration-with-medical-background-having-heart-beat-doctor-and-stethoscope.jpg')
    
    st.title('CHD Risk Prediction App')
    st.write('Enter the following information to predict your CHD risk:')
    age = st.slider('Age', 25, 80, 50)
    # Get the sex from the user
    sex = st.selectbox('Sex', ['Male', 'Female'])

    # Map the sex to the corresponding numeric value
    sex_val = 1 if sex == 'Male' else 0
    # Define the dictionary to map chest pain type to numeric values
    cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}

    # Get the chest pain type from the user
    cp = st.selectbox('Chest Pain Type', list(cp_dict.keys()))

    # Map the chest pain type to the corresponding numeric value using the dictionary
    cp_val = cp_dict[cp]
   
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 500, 240)
    
    # Get the Fasting Blood Sugar value from the user
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])

    # Map the Fasting Blood Sugar value to the corresponding numeric value
    fbs_val = int(fbs == 'True')
    
    # Define the dictionary to map Resting ECG to numeric values
    restecg_dict = {'Normal': 0, 'ST-T Abnormality': 1, 'Probable/Definite Left Ventricular Hypertrophy': 2}

    # Get the Resting ECG value from the user
    restecg = st.selectbox('Resting ECG', list(restecg_dict.keys()))

    # Map the Resting ECG value to the corresponding numeric value using the dictionary
    restecg_val = restecg_dict[restecg]
    
    thalach = st.slider('Maximum Heart Rate Achieved', 50, 220, 150)
    
    # Get the Exercise Induced Angina value from the user
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])

    # Map the Exercise Induced Angina value to the corresponding numeric value
    exang_val = int(exang == 'Yes')
    
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 2.0, 0.1)
    #slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    # Define the dictionary to map Slope of Peak Exercise ST Segment to numeric values
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}

    # Get the Slope of Peak Exercise ST Segment value from the user
    slope = st.selectbox('Slope of Peak Exercise ST Segment', list(slope_dict.keys()))

    # Map the Slope of Peak Exercise ST Segment value to the corresponding numeric value using the dictionary
    slope_val = slope_dict[slope]

    
    ca = int(st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3']))
    #thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # Define the dictionary to map Thalassemia to numeric values
    thal_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    # Get the Thalassemia value from the user
    thal = st.selectbox('Thalassemia', list(thal_dict.keys()))

    # Map the Thalassemia value to the corresponding numeric value using the dictionary
    thal_val = thal_dict[thal]
    
    features = np.array([[age,	sex_val,	cp_val,	trestbps,	chol,	fbs_val,	restecg_val,	thalach,	exang_val,	oldpeak,	slope_val,	ca,	thal_val]])
    prediction = predict_chd_risk(features)
    st.write('Your predicted CHD risk is:', prediction)
    st.write('array',features)  
    

    # set up the bar chart data
    if prediction == 0:
        values = [1-prediction, prediction]
        labels = ['Low Risk', 'High Risk']
    else:
        values = [1-prediction, prediction]
        labels = ['Low Risk', 'High Risk']

    # create the bar chart
    fig, ax = plt.subplots()
    ax.barh(labels, values)
    ax.set_xlabel('Probability')
    ax.set_ylabel('CHD Risk')

    # display the bar chart in Streamlit
    st.pyplot(fig)
    
    if prediction == 0:
        st.markdown(
            """
            <h2 style='text-align: center;'>Your predicted CHD risk is <span style='color: green;'>low</span>.</h2>
            <div style='text-align: center;'>
                <img src='https://media.giphy.com/media/2WRAeRpoIWU0wVxj7u/giphy.gif'>
            </div>
            """
            , unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <h2 style='text-align: center;'>Your predicted CHD risk is <span style='color: red;'>high</span>.</h2>
            <div style='text-align: center;'>
                <img src='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjZlNGI1YThlM2YyY2E2Nzg1N2JhOTBmMGU2YzQ5ZDM2YTk4MWU5OCZjdD1n/pme5OjYY04WaWD0WMX/giphy.gif'>
            </div>
            """
            , unsafe_allow_html=True
        )

if __name__ == "__main__":
    app()

    
