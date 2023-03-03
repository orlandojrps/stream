import streamlit as st
import pandas as pd
import folium
from geopy.geocoders import Nominatim
import time
import streamlit.components as stc
def geocode_and_plot_addresses(df):
    geolocator = Nominatim(user_agent='user-my-application') # create a geolocator object
    addresses = df['Address'].head(9).tolist() # extract addresses from DataFrame 
    lats = []
    longs = []
    
    # loop over addresses, geocode each one, and extract the latitude and longitude
    for address in addresses:
        location = geolocator.geocode(address)
        time.sleep(0) # add a 1.1-second interval between requests

    # create a table with the summary statistics
    num_houses = df.shape[0]
    mean_price = df['Price'].mean()
    min_price = df['Price'].min()
    max_price = df['Price'].max()

    data = {
        ' ': [stc.html('<i class="fas fa-home"></i>'), 
              stc.html('<i class="fas fa-money-bill-wave"></i>'), 
              stc.html('<i class="fas fa-arrow-down"></i>'), 
              stc.html('<i class="fas fa-arrow-up"></i>')],
        'Statistic': ['Number of Houses', 'Average Price', 'Lowest Price', 'Highest Price'],
        'Value': [num_houses, f'${mean_price:,.2f}', f'${min_price:,.2f}', f'${max_price:,.2f}']
    }

    table = stc.html('<i class="fas fa-table"></i>') + st.table(data)

    # plot the coordinates on a map using Folium
    map_center = [51.897928, -8.470579] # center the map on Cork City
    m = folium.Map(location=map_center, zoom_start=12)
    for i, row in df.iterrows():
        if row['Latitude'] and row['Longitude']:
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Address']).add_to(m)

    # add the table and map to the interface
    stc.html('<hr>') # add a horizontal line
    stc.html('<h2>Summary Statistics:</h2>')
    stc.html('<br>')
    stc.html('<br>')
    st.write(table, unsafe_allow_html=True)
    stc.html('<hr>') # add a horizontal line
    stc.html('<h2>Map:</h2>')
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

