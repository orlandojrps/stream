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
Created with ❤️ by [Data Professor](https://youtube.com/dataprofessor/).
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
col1, col2,col3, col5,col6 = st.columns(5)
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
metric_html = f"<div style='font-size: 18px; font-weight: bold;'>Min (m2):  {min_area:}</div>"
metric_html2 = f"<div style='font-size: 18px; font-weight: bold;'>Max (m2):  {max_area:}</div>"

col1m, col2m = st.columns(2)
col2.markdown(metric_html + metric_html2, unsafe_allow_html=True)

##################################


col3.metric("Humidity", "86%", "4%")
col5.metric("Humidity", "86%", "4%")
col6.metric("Humidity", "86%", "4%")






# Row B
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns((7,3))
with c1:
    st.markdown('### Heatmap')
    plost.time_hist(
    data=seattle_weather,
    date='date',
    x_unit='week',
    y_unit='day',
    color=time_hist_color,
    aggregate='median',
    legend=None,
    height=345,
    use_container_width=True)
with c2:
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)

# Row C
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)

#!streamlit run /content/app.py &>/content/logs.txt &

#!npx localtunnel --port 8501













'''
# Filtrando dataframe df com base em df_city_area selecionado
df_filtered = df[df['city_area'] == "Cork City North East"]

# Calculando média do campo df.price para o dataframe filtrado
avg_price = df_filtered['Price'].mean()
#st.markdown(f"Average Price: ${avg_price:,.2f}")

#df_filtered.describe()

'''
import streamlit as st
import pandas as pd
import folium
from geopy.geocoders import Nominatim
import time
from PIL import Image

def geocode_and_plot_addresses(df):
    geolocator = Nominatim(user_agent='user-my-application') # create a geolocator object
    addresses = df['Address'].head(9).tolist() # extract addresses from DataFrame 
    lats = []
    longs = []
    
    # loop over addresses, geocode each one, and extract the latitude and longitude
    for address in addresses:
        location = geolocator.geocode(address)
        time.sleep(0) # add a 1.1-second interval between requests
    
    # calculate summary statistics
    num_houses = df['Price'].count()
    avg_price = df['Price'].mean()
    lowest_price = df['Price'].min()
    highest_price = df['Price'].max()
    
    # create a table with the summary statistics
    summary_table = pd.DataFrame({
        'Number of Houses': [num_houses],
        'Average Price': [avg_price],
        'Lowest Price': [lowest_price],
        'Highest Price': [highest_price]
    })
    
    # create a streamlit table and add it to the interface
    styler = summary_table.style.format({
        'Average Price': '${:.2f}',
        'Lowest Price': '${:.2f}',
        'Highest Price': '${:.2f}'
    })

    house_icon = Image.open("https://raw.githubusercontent.com/orlandojrps/stream/0e63fdd22c7dfa974e221319fb4ca60acc174d8f/teste.jpg")

    st.write(styler, unsafe_allow_html=True)
    
    # plot the coordinates on a map using Folium
    map_center = [51.897928, -8.470579] # center the map on Cork City
    m = folium.Map(location=map_center, zoom_start=12)
    for i, row in df.iterrows():
        if row['Latitude'] and row['Longitude']:
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Address']).add_to(m)

    # add the map to the interface
    st.write(m._repr_html_(), unsafe_allow_html=True)

# Load the DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

# Create input to select city_area
city_area = st.selectbox('Select a city_area', df['city_area'].unique())

# Filter the DataFrame based on the selected city_area
filtered_df = df[df['city_area'] == city_area]

# Create a button to plot addresses on the map
if st.button('Plot addresses on the map'):
    geocode_and_plot_addresses(filtered_df)
