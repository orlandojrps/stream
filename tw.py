import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import geopy.geocoders
import folium
from streamlit_folium import folium_static

geopy.geocoders.options.default_user_agent = "my-application"
geolocator = geopy.geocoders.Nominatim(user_agent=geopy.geocoders.options.default_user_agent)

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

def get_location(tweet):
    try:
        location = geolocator.geocode(tweet)
        if location is not None:
            return location.address.split(",")[-1].strip()
        else:
            return None
    except geopy.exc.GeocoderTimedOut:
        return None

st.title("Twitter Sentiment Analysis")

search_query = st.text_input("Enter a search query:")

if st.button("Search"):
    st.write(f"Fetching tweets for {search_query}...")
    tweets = get_tweets(search_query, 100)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    df['Sentiment'] = df['Tweet'].apply(get_sentiment)
    df['Location'] = df['Tweet'].apply(get_location)
    sentiment_counts = df['Sentiment'].value_counts()
    st.write(f"Found {len(tweets)} tweets.")
    st.write(f"Positive tweets: {sentiment_counts['positive']}")
    st.write(f"Negative tweets: {sentiment_counts['negative']}")
    st.bar_chart(sentiment_counts)

    # create a map
    m = folium.Map(location=[0,0], zoom_start=2)

    # get the location counts for each sentiment
    location_counts = {}
    for i, row in df.iterrows():
        if row['Location'] is not None:
            if row['Sentiment'] not in location_counts:
                location_counts[row['Sentiment']] = {}
            if row['Location'] not in location_counts[row['Sentiment']]:
                location_counts[row['Sentiment']][row['Location']] = 0
            location_counts[row['Sentiment']][row['Location']] += 1

    # add the markers to the map
    for sentiment in location_counts:
        for location in location_counts[sentiment]:
            count = location_counts[sentiment][location]
            try:
                location_info = geolocator.geocode(location)
                lat, lon = location_info.latitude, location_info.longitude
                tooltip = f"{location} ({count} {sentiment})"
                folium.Marker(location=[lat, lon], tooltip=tooltip, icon=folium.Icon(color='green' if sentiment=='positive' else 'red')).add_to(m)
            except Exception as e:
                print(e)

    folium_static(m)

    st.write(df)
