# Importar bibliotecas
import streamlit as st
import folium
from geopy.geocoders import Nominatim
import time
import pandas as pd

# Função para plotar mapa

def geocode_and_plot_addresses(df):
    geolocator = Nominatim(user_agent='user-my-application') # create a geolocator object
    addresses = df['Address'].tolist() # extract addresses from DataFrame
    lats = []
    longs = []
    
    # loop over addresses, geocode each one, and extract the latitude and longitude
    for address in addresses:
        location = geolocator.geocode(address)
        time.sleep(0) # add a 1.1-second interval between requests

    
    # plot the coordinates on a map using Folium
    map_center = [51.897928, -8.470579] # center the map on Cork City
    m = folium.Map(location=map_center, zoom_start=12)
    for i, row in df.iterrows():
        if row['Latitude'] and row['Longitude']:
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Address']).add_to(m)
    return m

# Carregar DataFrame
#df2 = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/stage2_df.csv')
df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

# Criar input para selecionar city_area
city_area = st.selectbox('Selecione uma city_area', df['city_area'].unique())

# Filtrar DataFrame com base na city_area selecionada
filtered_df = df[df['city_area'] == city_area]

# Criar botão para plotar endereços no mapa
if st.button('Plotar endereços no mapa'):
    map = geocode_and_plot_addresses(filtered_df.head(9))
    # Adicionar mapa na interface
    st.write(map._repr_html_(), unsafe_allow_html=True)
