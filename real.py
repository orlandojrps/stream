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

    house_icon = Image.open("https://raw.githubusercontent.com/orlandojrps/stream/0e63fdd22c7dfa974e221319fb4ca60acc174d8f/teste.jpg").resize((32, 32))
    price_icon = Image.open("https://raw.githubusercontent.com/orlandojrps/stream/0e63fdd22c7dfa974e221319fb4ca60acc174d8f/teste.jpgg").resize((32, 32))

    styler.add_rows([
        ['<img src="https://raw.githubusercontent.com/orlandojrps/stream/0e63fdd22c7dfa974e221319fb4ca60acc174d8f/teste.jpg,{}"/> Number of Houses'.format(house_icon), num_houses]
    ])

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
