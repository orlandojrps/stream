# Importar bibliotecas
import streamlit as st
import folium
from geopy.geocoders import Nominatim
import time
import pandas as pd

# Função para plotar mapa
def geocode_and_plot_addresses(df):
    geolocator = Nominatim(user_agent='meu-novo-user-agent')
    addresses = df['Address'].tolist() # extrair endereços do DataFrame
    lats = []
    longs = []
    
    # loop sobre endereços, geocodificar cada um, e extrair a latitude e longitude
    for address in addresses:
        location = geolocator.geocode(address)
        time.sleep(2) # adicionar um intervalo de 2 segundos entre as solicitações
    
        if location:
            lats.append(location.latitude)
            longs.append(location.longitude)
        else:
            lats.append(None)
            longs.append(None)
    
    # adicionar os marcadores ao mapa usando Folium
    map_center = [51.897928, -8.470579] # centralizar o mapa em Cork City
    m = folium.Map(location=map_center, zoom_start=12)
    for i, row in df.iterrows():
        if lats[i] and longs[i]:
            folium.Marker([lats[i], longs[i]], popup=row['Address']).add_to(m)
    
    return m

# Carregar DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/orlandojrps/stream/main/df_final.csv')

# Criar input para selecionar city_area
city_area = st.selectbox('Selecione uma city_area', df['city_area'].unique())

# Filtrar DataFrame com base na city_area selecionada
filtered_df = df[df['city_area'] == city_area]

# Criar botão para plotar endereços no mapa
if st.button('Plotar endereços no mapa'):
    map = geocode_and_plot_addresses(filtered_df)
    # Adicionar mapa na interface
    st.write(map._repr_html_(), unsafe_allow_html=True)
