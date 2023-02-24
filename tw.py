import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import geopy
import geopy.exc
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import plotly.express as px

geolocator = Nominatim(user_agent="streamlit_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

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
    if location is not None:
        try:
            country = geolocator.reverse(location, exactly_one=True).raw['address']['country']
        except (geopy.exc.GeocoderTimedOut, geopy.exc.GeocoderServiceError, KeyError):
            country = None
    else:
        country = None
    return country

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
    country_sentiment_counts = df.groupby('Country')['Sentiment'].value_counts().unstack(fill_value=0)
    country_sentiment_counts['Total'] = country_sentiment_counts.sum(axis=1)
    country_sentiment_counts = country_sentiment_counts.reset_index()
    fig = px.choropleth(country_sentiment_counts, locations="Country", color="Total", 
                        hover_name="Country", projection="natural earth")
    st.plotly_chart(fig)
