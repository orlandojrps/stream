# -*- coding: utf-8 -*-
"""Copy of Untitled52.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MhDY33gYFbZ09l94hShl_6-hgqwqkyZa
"""

import streamlit as st
import pandas as pd
import folium
from geopy.geocoders import Nominatim
import time
from PIL import Image
import plost
import streamlit.components.v1 as components
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#!pip install plost

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

#%%writefile app.py
 
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Real Estate Market `Cork City v. 0.1`')
st.sidebar.image("https://raw.githubusercontent.com/orlandojrps/stream/main/map.jpg", use_column_width=True)

# Add All option
city_areas = list(df['city_area'].unique())
city_areas.insert(0, "ALL")

# Area Select
#st.sidebar.subheader('Area Filter')
st.sidebar.markdown(
    f'<h3 style="color: #1B9685;">Area Filter</h3>',
    unsafe_allow_html=True
)
df_city_area = st.sidebar.selectbox('Select City Area', city_areas)


# Filtrando o dataframe com base na área selecionada
#if df_city_area == "ALL":
 #   filtered_df = df[df['city_area'] == "Cork City South Central"]
#else:
 #   filtered_df = df[df['city_area'] == df_city_area]
#value = "\U0001F3E1"
#st.markdown(f"<div style='font-size: 64px;'>{value}</div>", unsafe_allow_html=True)

st.sidebar.markdown('''---''')   
#st.sidebar.subheader('Price Prediction Modeling \U0001F3AF')
st.sidebar.markdown(
    f'<h3 style="color: #1B9685;">Price Prediction Modeling \U0001F3AF</h3>',
    unsafe_allow_html=True
)
pred_city_area = st.sidebar.selectbox('City Area', city_areas)

#st.sidebar.subheader('Price Prediction Modeling')
pred_area = st.sidebar.slider('Specify Size', 40, 800, 150)


beds_list = sorted(df["Beds"].unique())

baths_list = sorted(df["Baths"].unique())

#st.sidebar.subheader('Donut chart parameter')
pred_beds = st.sidebar.selectbox('Qty Beds', beds_list)

#st.sidebar.subheader('Donut chart parameter')
pred_baths = st.sidebar.selectbox('Qty Baths', baths_list)

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created by Orlando).
''')


# Filtrando dataframe df com base em df_city_area selecionado
print(df_city_area)
if df_city_area == "ALL":
    df_filtered = df[df['city_area'] != ""]
    
else:
    df_filtered = df[df['city_area'] == df_city_area]

# Calculando média do campo df.price para o dataframe filtrado
avg_price = df_filtered['Price'].mean()

# Calculando média do campo df.area para o dataframe filtrado
avg_area = df_filtered['Area'].mean()

# Calculando minimo para o dataframe filtrado
min_price = df_filtered['Price'].min()

# Calculando max para o dataframe filtrado
max_price = df_filtered['Price'].max()

# Calculando minimo para o dataframe filtrado
min_area = df_filtered['Area'].min()

# Calculando max para o dataframe filtrado
max_area = df_filtered['Area'].max()

# Calculando média do campo df.area para o dataframe filtrado
avg_beds = df_filtered['Beds'].mean()

# Calculando minimo para o dataframe filtrado
min_beds = df_filtered['Beds'].min()

# Calculando max para o dataframe filtrado
max_beds = df_filtered['Beds'].max()

# Calculando média do campo df.area para o dataframe filtrado
avg_baths = df_filtered['Baths'].mean()

# Calculando minimo para o dataframe filtrado
min_baths = df_filtered['Baths'].min()

# Calculando max para o dataframe filtrado
max_baths = df_filtered['Baths'].max()

# Calculando count
n_houses = df_filtered['Baths'].count()

st.markdown(
    """
    <style>
        .main {
            padding-top: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Row A
st.markdown('### Metrics: '+ df_city_area)

col0,col1, col2,col3, col4 = st.columns(5)
metric_html0 = f"<div style='font-size: 24px; font-weight: bold;'>Average Price:</div>"
col1.markdown(metric_html0, unsafe_allow_html=True)
col1.metric("", f"€ {avg_price:,.2f}", " ")

#col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min Price: € {min_price:,.2f}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max Price: € {max_price:,.2f}</div>"

#col1m, col2m = st.columns(2)
col1.markdown(metric_html + metric_html2, unsafe_allow_html=True)

#fig, ax = plt.subplots(figsize=(6,4))
#sns.kdeplot(data=df_filtered, x="Price", ax=ax, shade=True, color="#1f77b4", alpha=0.8)
#ax.set(xlabel='Price', ylabel='Density')
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.tick_params(axis='both', which='both', length=0)
#col1.pyplot(fig)


# Calculate the average price
avg_price = df_filtered['Price'].mean()

# Create the plot
fig, ax = plt.subplots(figsize=(6,4))
sns.kdeplot(data=df_filtered, x="Price", ax=ax, shade=True, color="#1f77b4", alpha=0.8)
ax.set(xlabel='Price', ylabel='Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

# Add a vertical line at the average price
ax.axvline(avg_price, color='red', linestyle='--', linewidth=2)

# Display the plot
col1.pyplot(fig)


###############################

metric_col2 = f"<div style='font-size: 24px; font-weight: bold;'>Average Area (m2):</div>"
col2.markdown(metric_col2, unsafe_allow_html=True)
col2.metric("", f" {avg_area:,.2f}", " ")
#col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (m2):  {min_area:,.2f}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (m2):  {max_area:,.2f}</div>"

#col1m, col2m = st.columns(2)
col2.markdown(metric_html + metric_html2, unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(6,4))
sns.boxplot(data=df_filtered, x="Area", ax=ax, color="#1f77b4", linewidth=1)

# Remove spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

# Add labels
ax.set(xlabel='Area', ylabel=df_city_area)

# Display the plot
col2.pyplot(fig)

##################################


###############################

metric_col3 = f"<div style='font-size: 24px; font-weight: bold;'>Average Beds (Qty):</div>"
col3.markdown(metric_col3, unsafe_allow_html=True)
col3.metric("", f" {avg_beds:,.2f}", " ")
#col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (Qty):  {min_beds:}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (Qty):  {max_beds:}</div>"

#col1m, col2m = st.columns(2)
col3.markdown(metric_html + metric_html2, unsafe_allow_html=True)


# Create the scatterplot
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df_filtered, x="Area", y="Price", hue="Beds", palette="deep", alpha=0.8)

# Add a title and axis labels
plt.title("Scatterplot of Price vs. Area (colored by Beds)")
plt.xlabel("Area")
plt.ylabel("Price")

# Remove top and right spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

# Display the plot
col3.pyplot(fig)


##################################
###############################

metric_col4 = f"<div style='font-size: 24px; font-weight: bold;'>Average Baths (Qty):</div>"
col4.markdown(metric_col4, unsafe_allow_html=True)
col4.metric("", f" {avg_baths:,.2f}", " ")
#col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (Qty):  {min_baths:}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (Qty):  {max_baths:}</div>"

#col1m, col2m = st.columns(2)
col4.markdown(metric_html + metric_html2, unsafe_allow_html=True)




# Create the bar plot
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(data=df_filtered, x="Baths", palette="deep")

# Add a title and axis labels
plt.title("Count of Bathrooms")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")

# Remove top and right spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

# Display the plot
col4.pyplot(fig)

##################################
###############################

#metric_col0 = f"<div style='font-size: 24px; font-weight: bold; color:#000000;'>Number Houses (Qty):</div>"
#col0.markdown(metric_col0, unsafe_allow_html=True)
#col0.metric("", f" {n_houses:}", " ")
 

    
    
    
    
metric_col0 = "<div style='font-size: 24px; font-weight: bold;'>Number Houses:</div>"
col0.markdown(metric_col0, unsafe_allow_html=True)

# Adicionar o valor de n_houses e o ícone da casa
value = f"{n_houses} \U0001F3E1"
col0.markdown(f"<div style='font-size: 64px;'>{value}</div>", unsafe_allow_html=True)


 
##################################





###############################
#col5,col6, col7,col8, col9 = st.columns(5)
#metric_col5 = f"<div style='font-size: 24px; font-weight: bold;'>Average Area (m2):</div>"
#col5.markdown(metric_col5, unsafe_allow_html=True)
#col5.metric("", f" {avg_area:,.2f}", " ")




# Generate sample data
data = np.random.randn(100)

# Define histogram settings
num_bins = 100
#hist_range = (-3, 3)

# Create histogram and display in col5
#col5, col6, col7, col8, col9 = st.columns(5)
#col5.header("Histogram Example")
#hist_values, hist_edges = np.histogram(df_filtered['Price'], bins=num_bins)
#col5.bar_chart(hist_values, width=200, height=200, use_container_width=False)

#8888888888888888888888888888888888888888
#col5, col6, col7, col8, col9 = st.columns(5)
#fig, ax = plt.subplots(figsize=(6,4))
#sns.kdeplot(data=df_filtered, x="Price", ax=ax, shade=True, color="#1f77b4", alpha=0.8)
#ax.set(xlabel='Price', ylabel='Density')
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.tick_params(axis='both', which='both', length=0)
#col5.pyplot(fig)
#888888888888888888888888888888888888888 


#col5.set(xlabel='Price', ylabel='Density')
#col1m, col2m = st.columns(2)
#metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (m2):  {min_area:,.2f}</div>"
#metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (m2):  {max_area:,.2f}</div>"

#col1m, col2m = st.columns(2)
#col2.markdown(metric_html + metric_html2, unsafe_allow_html=True)

##################################












#col3.metric("Humidity", "86%", "4%")
#col5.metric("Humidity", "86%", "4%")
#col6.metric("Humidity", "86%", "4%")



# Row B

seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns(2)
with c1:
    st.markdown('### Map Plot')
    
def geocode_and_plot_addresses(df):
    geolocator = Nominatim(user_agent='user-my-application') # create a geolocator object
    addresses = df['Address'].head(9).tolist() # extract addresses from DataFrame 
    lats = []
    longs = []
    
    # loop over addresses, geocode each one, and extract the latitude and longitude
    for address in addresses:
        location = geolocator.geocode(address)
        time.sleep(0) # add a 1.1-second interval between requests
    
    # plot the coordinates on a map using Folium
    map_center = [51.897928, -8.470579] # center the map on Cork City
    #m = folium.Map(location=map_center, zoom_start=12)
    m = folium.Map(location=map_center, zoom_start=12, height='80%', width='100%')
    for i, row in df.iterrows():
        if row['Latitude'] and row['Longitude']:
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Address']).add_to(m)

    # add the map to the interface
    c1.write(m._repr_html_(), unsafe_allow_html=True)

# Load the DataFrame
#df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

# Create input to select city_area
#city_area = st.selectbox('Select a city_area', df['city_area'].unique())

# Filter the DataFrame based on the selected city_area
#filtered_df = df[df['city_area'] == city_area]





# Create a button to plot addresses on the map
#if (0==0):
 #   geocode_and_plot_addresses(df_filtered)
  # Create a button to plot addresses on the map




if c1.button('Plot addresses on the map'):
    geocode_and_plot_addresses(df_filtered)  

    
    
    
    
    
    
    
with c2:
    st.markdown('### Best Offers')
    
map_url = 'https://www.instantstreetview.com/@51.902544,-8.478546,-7.89h,0p,0z,t3TYjiEEK68_KwPyXNtsJA'
st.write(f'<a href="{map_url}" target="_blank">Click here to view map</a>', unsafe_allow_html=True)
     
metric_c2 = "<div style='font-size: 24px; font-weight: bold;'>Number Houses:</div>"
c2.markdown(metric_c2, unsafe_allow_html=True)

# Adicionar o valor de n_houses e o ícone da casa
value = f"{n_houses} \U0001F3E1"
c2.markdown(f"<div style='font-size: 64px;'>{value}</div>", unsafe_allow_html=True)

# Get the top 10 rows with the lowest prices
df_filtered_links = df_filtered.nsmallest(10, "Price")

# Create a list of clickable links for the top 10 rows
link_list = []
for index, row in df_filtered_links.iterrows():
    link = f'<a href="{row["link"]}" target="_blank">{row["Address"]}</a>'
    link_list.append(link)

# Display the list of clickable links
for link in link_list:
    c2.markdown(link, unsafe_allow_html=True)
    




 
import webbrowser

# Function to create popup window with link to Google Street View
def street_view_popup():
    street_view_url = f"https://www.instantstreetview.com/@51.902544,-8.478546,-7.89h,0p,0z,t3TYjiEEK68_KwPyXNtsJA"
    webbrowser.open_new_tab(street_view_url)

 
# Add button for each row to open Street View
 
    st.button(
        label=f"View Street View of location",
        on_click= street_view_popup()
    )





#st.write(html_component)






# Row C
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)



################## PREDICTION#################################
from io import BytesIO
import requests
mLink = 'https://github.com/orlandojrps/stream/blob/main/model_lr_real_estate.pkl?raw=true'
mfile = BytesIO(requests.get(mLink).content)

model = joblib.load(mfile)

print(model)

# Load the pre-trained linear regression model
lr_model = joblib.load(mfile)
 
# Define the function to make a prediction
def predict(features):
    prediction = lr_model.predict(features)
    return prediction[0]
    
    
df_filtered_pred = df[df['city_area'] == pred_city_area]
    
    
    # Calculate the price per square meter
df_filtered_pred['Price per m2'] = df_filtered_pred['Price'] / df_filtered_pred['Area']
priceM2 = df_filtered_pred['Price'] / df_filtered_pred['Area']

# calculate the median per square meter per city area
median_price_m2 = df_filtered_pred.groupby('city_area')['Price per m2'].median()

# Calculate the average price per square meter by city area
mean_price_m2 = df_filtered_pred.groupby('city_area')['Price per m2'].mean()

# Calculate the average price by city area
average_price = df_filtered_pred.groupby('city_area')['Price'].mean()

# Add the mean price per square meter and average price by city area as new columns to the DataFrame
df_filtered_pred['Mean Price per m2'] = df_filtered_pred['city_area'].map(mean_price_m2)
priceMean=df_filtered_pred['city_area'].map(mean_price_m2)
df_filtered_pred['Average Price by City Area'] = df_filtered_pred['city_area'].map(average_price)
priceArea = df_filtered_pred['city_area'].map(average_price)
    

features = np.array([pred_beds, pred_baths, mean_price_m2, mean_price_m2, mean_price_m2, mean_price_m2]).reshape(1, -1)
    
#features = np.array([pred_beds, pred_baths, pred_area, median_price_m2, mean_price_m2, average_price]).reshape(1, -1)
#######features = np.array([pred_beds, pred_baths, pred_area, pred_area, pred_area, pred_area]).reshape(1, -1)
prediction = predict(features)
#c2.write('Your Suggested Price is:', 12)  
c2.markdown(f"<div style='font-size: 64px;'>{priceArea}</div>", unsafe_allow_html=True)
