import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
from geopy.geocoders import Nominatim
import plotly.express as px

geolocator = Nominatim(user_agent="my_app")

def get_tweets(query, limit):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if tweet.lang == 'en':
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.username, tweet.content, tweet.location])
    return tweets

def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return 'positive' if analysis.sentiment.polarity >= 0 else 'negative'

st.title("Twitter Sentiment Analysis")

search_query = st.text_input("Enter a search query:")

if st.button("Search"):
    st.write(f"Fetching tweets for {search_query}...")
    tweets = get_tweets(search_query, 100)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Location'])
    df['Sentiment'] = df['Tweet'].apply(get_sentiment)
    sentiment_counts = df['Sentiment'].value_counts()
    st.write(f"Found {len(tweets)} tweets.")
    st.write(f"Positive tweets: {sentiment_counts['positive']}")
    st.write(f"Negative tweets: {sentiment_counts['negative']}")
    st.bar_chart(sentiment_counts)

    # Geocode the location of each tweet to get the country
    df['Country'] = df['Location'].apply(lambda x: geolocator.geocode(x, exactly_one=True, timeout=10).raw['address']['country'] if x is not None else None)
    
    # Aggregate the data by country and sentiment
    df_agg = df.groupby(['Country', 'Sentiment'], as_index=False).agg({'Tweet': 'count'})

    # Pivot the table to have one row per country and columns for positive and negative sentiment counts
    df_pivot = df_agg.pivot(index='Country', columns='Sentiment', values='Tweet').fillna(0)
    df_pivot['Total'] = df_pivot['positive'] + df_pivot['negative']

    # Plot the data on a world map
    fig = px.choropleth(df_pivot, locations=df_pivot.index, color='Total', scope='world', hover_data=['positive', 'negative'])
    st.plotly_chart(fig, use_container_width=True)

    st.write(df)
