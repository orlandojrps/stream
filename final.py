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

#!pip install plost

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

#%%writefile app.py
 
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Real Estate Market `Cork City v. 0.1`')
st.sidebar.image("https://raw.githubusercontent.com/orlandojrps/stream/main/map.jpg", use_column_width=True)

# Adicionando filtro para selecionar df.city_area
df_city_area = st.sidebar.selectbox('Select City Area', df['city_area'].unique())


st.sidebar.subheader('Heat map parameter')
time_hist_color = st.sidebar.selectbox('Color by', ['temp_min', 'temp_max', 'city_area']) 



st.sidebar.subheader('Donut chart parameter')
donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created by Orlando).
''')


# Filtrando dataframe df com base em df_city_area selecionado
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
col1, col2,col3, col4,col6 = st.columns(5)
metric_html0 = f"<div style='font-size: 24px; font-weight: bold;'>Average Price:</div>"
col1.markdown(metric_html0, unsafe_allow_html=True)
col1.metric("", f"€ {avg_price:,.2f}", " ")

col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min Price: € {min_price:,.2f}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max Price: € {max_price:,.2f}</div>"

col1m, col2m = st.columns(2)
col1.markdown(metric_html + metric_html2, unsafe_allow_html=True)

###############################

metric_col2 = f"<div style='font-size: 24px; font-weight: bold;'>Average Area (m2):</div>"
col2.markdown(metric_col2, unsafe_allow_html=True)
col2.metric("", f" {avg_area:,.2f}", " ")
col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (m2):  {min_area:,.2f}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (m2):  {max_area:,.2f}</div>"

col1m, col2m = st.columns(2)
col2.markdown(metric_html + metric_html2, unsafe_allow_html=True)

##################################


###############################

metric_col3 = f"<div style='font-size: 24px; font-weight: bold;'>Average Beds (Qty):</div>"
col3.markdown(metric_col3, unsafe_allow_html=True)
col3.metric("", f" {avg_beds:,.2f}", " ")
col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (Qty):  {min_beds:}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (Qty):  {max_beds:}</div>"

col1m, col2m = st.columns(2)
col3.markdown(metric_html + metric_html2, unsafe_allow_html=True)

##################################
###############################

metric_col4 = f"<div style='font-size: 24px; font-weight: bold;'>Average Baths (Qty):</div>"
col4.markdown(metric_col4, unsafe_allow_html=True)
col4.metric("", f" {avg_baths:,.2f}", " ")
col1m, col2m = st.columns(2)
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (Qty):  {min_baths:}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (Qty):  {max_baths:}</div>"

col1m, col2m = st.columns(2)
col4.markdown(metric_html + metric_html2, unsafe_allow_html=True)

##################################






#col3.metric("Humidity", "86%", "4%")
#col5.metric("Humidity", "86%", "4%")
#col6.metric("Humidity", "86%", "4%")



# Row B

seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns(2)
with c1:
    st.markdown('### Heatmap')
    
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
    m = folium.Map(location=map_center, zoom_start=13, height=400, width=400)
    for i, row in df.iterrows():
        if row['Latitude'] and row['Longitude']:
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Address']).add_to(m)

    # add the map to the interface
    st.write(m._repr_html_(), unsafe_allow_html=True)

# Load the DataFrame
#df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

# Create input to select city_area
#city_area = st.selectbox('Select a city_area', df['city_area'].unique())

# Filter the DataFrame based on the selected city_area
#filtered_df = df[df['city_area'] == city_area]

# Create a button to plot addresses on the map
if st.button('Plot addresses on the map'):
    geocode_and_plot_addresses(df_filtered)
    
    
with c2:
    st.markdown('### Donut chart')
    #plost.donut_chart(
     #   data=stocks,
      #  theta=donut_theta,
       # color='company',
        #legend='bottom', 
        #use_container_width=True)

#ap_url = 'https://www.google.com/maps/@51.9011024,-8.4941951,3a,75y,254.1h,90t/data=!3m7!1e1!3m5!1sAF1QipOz-BpF13JhlgTmgHuyBHQE7VnJv6uNE7-UUfL6!2e10!3e12!7i16384!8i8192'
#map_url = 'https://maps.google.com/maps?q=Tangesir%20Dates%20Products&amp;t=&amp;z=13&amp;ie=UTF8&amp;iwloc=&amp;output=embed'
map_url = 'https://www.instantstreetview.com/@51.902544,-8.478546,-7.89h,0p,0z,t3TYjiEEK68_KwPyXNtsJA'

st.write(f'<iframe src="{map_url}" width="1000" height="500"></iframe>', unsafe_allow_html=True)
map_url = 'https://www.google.com/maps/@51.9011024,-8.4941951,3a,75y,254.1h,90t/data=!3m7!1e1!3m5!1sAF1QipOz-BpF13JhlgTmgHuyBHQE7VnJv6uNE7-UUfL6!2e10!3e12!7i16384!8i8192'

st.write(f'<a href="{map_url}" target="_blank">Click here to view map</a>', unsafe_allow_html=True)

#<iframe src="https://maps.google.com/maps?q=Tangesir%20Dates%20Products&amp;t=&amp;z=13&amp;ie=UTF8&amp;iwloc=&amp;output=embed" width=300 height=150 allowfullscreen></iframe>


from streamlit.components.v1 import html

google_maps_url = "https://www.google.com/maps/embed/v1/place?key=<YOUR_API_KEY>&q=Cork+City"
iframe = f'<iframe src="https://www.google.com/maps/@51.9011024,-8.4941951,3a,75y,254.1h,90t/data=!3m7!1e1!3m5!1sAF1QipOz-BpF13JhlgTmgHuyBHQE7VnJv6uNE7-UUfL6!2e10!3e12!7i16384!8i8192" width="1000" height="500"></iframe>'
html_component = html.Iframe(src=iframe, width=1000, height=500)
st.markdown("<h1>Google Maps Embedded</h1>", unsafe_allow_html=True)
st.components.v1.html(html_component)




# Row C
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)

 
