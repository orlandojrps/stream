import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
import transformers
import torch

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
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer.encode_plus(
        tweet,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    outputs = model(inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])
    prediction = torch.argmax(outputs[0]).item()
    labels = ['negative', 'neutral', 'positive']
    return labels[prediction]

st.title("Twitter Sentiment Analysis")

search_query = st.text_input("Enter a search query:")

if st.button("Search"):
    st.write(f"Fetching tweets for {search_query}...")
    tweets = get_tweets(search_query, 100)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    df['Sentiment'] = df['Tweet'].apply(get_sentiment)
    sentiment_counts = df['Sentiment'].value_counts()
    st.write(f"Found {len(tweets)} tweets.")
    st.write(f"Positive tweets: {sentiment_counts['positive']}")
    st.write(f"Negative tweets: {sentiment_counts['negative']}")
    st.bar_chart(sentiment_counts)
    st.write(df)
