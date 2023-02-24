import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
from geopy.geocoders import Nominatim
import folium

def get_tweets(query, limit):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if tweet.lang == 'en':
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.username, tweet.content])
    return tweets

def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return 'positive' if analysis.sentiment.polarity >= 0 else 'negative'

def get_country(location):
    geolocator = Nominatim(user_agent="sentiment_analysis")
    location = geolocator.geocode(location, timeout=10)
    return location.address.split(",")[-1].strip()

st.title("Twitter Sentiment Analysis")

search_query = st.text_input("Enter a search query:")

if st.button("Search"):
    st.write(f"Fetching tweets for {search_query}...")
    tweets = get_tweets(search_query, 100)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    df['Sentiment'] = df['Tweet'].apply(get_sentiment)
    df['Country'] = df['User'].apply(get_country)
    sentiment_counts = df['Sentiment'].value_counts()
    st.write(f"Found {len(tweets)} tweets.")
    st.write(f"Positive tweets: {sentiment_counts['positive']}")
    st.write(f"Negative tweets: {sentiment_counts['negative']}")
    st.bar_chart(sentiment_counts)

    # Create a dataframe with the count of positive and negative tweets by country
    country_counts = df.groupby(['Country', 'Sentiment']).size().reset_index(name='Count')
    positive_counts = country_counts[country_counts['Sentiment'] == 'positive']
    negative_counts = country_counts[country_counts['Sentiment'] == 'negative']

    # Create a map with markers for positive and negative tweet counts
    st.subheader("Positive and Negative Tweet Counts by Country")
    map_center = [0, 0]
    m = folium.Map(location=map_center, zoom_start=1)
    for index, row in positive_counts.iterrows():
        country = row['Country']
        count = row['Count']
        location = geolocator.geocode(country, timeout=10)
        if location:
            lat = location.latitude
            lon = location.longitude
            tooltip_text = f"{country}: {count} positive tweets"
            folium.Marker(location=[lat, lon], tooltip=tooltip_text, icon=folium.Icon(color='green')).add_to(m)
    for index, row in negative_counts.iterrows():
        country = row['Country']
        count = row['Count']
        location = geolocator.geocode(country, timeout=10)
        if location:
            lat = location.latitude
            lon = location.longitude
            tooltip_text = f"{country}: {count} negative tweets"
            folium.Marker(location=[lat, lon], tooltip=tooltip_text, icon=folium.Icon(color='red')).add_to(m)
    folium_static(m)

    st.write(df)

