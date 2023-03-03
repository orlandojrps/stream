import streamlit as st
import pandas as pd
import folium
from geopy.geocoders import Nominatim
import time

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
    summary_table = pd.DataFrame({'Number of Houses': [num_houses],'Average Price': [avg_price],'Lowest Price': [lowest_price],'Highest Price': [highest_price]})
    
    # create a streamlit table and add it to the interface
    summary_table = pd.DataFrame({
        ' ': [st.image('https://img.icons8.com/material-outlined/24/000000/house.png')],
        'Number of Houses': [num_houses],
        ' ': [st.image('https://img.icons8.com/material-outlined/24/000000/price-tag.png')],
        'Average Price': [avg_price],
        ' ': [st.image('https://img.icons8.com/material-outlined/24/000000/price.png')],
        'Lowest Price': [lowest_price],
        ' ': [st.image('https://img.icons8.com/material-outlined/24/000000/high-price.png')],
        'Highest Price': [highest_price]
    })
    
    
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
