import time

import geocoder
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
# from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import random

import nltk
nltk.downloader.download('vader_lexicon')

# beareer = AAAAAAAAAAAAAAAAAAAAABBQPQEAAAAAmsn7HGaleQTElAetFKdr7W%2BPdCk%3DAgEYLIUXSKpBGYs19PIgSAov1C5ypX6xxFM1QmAnv8JoTete1j

# Authentication
consumerKey = "W3cEabcwQkp7evTrmP5XoHASj"
consumerSecret = "oEaUTl3BkL0sj1InihbWCEO4PQahAxhpUvrLWcU8gL74PGFN3x"
accessToken = "1087783552949702657-QL4T6ZruVPj3JE91HxfByGfDun6wGV"
accessTokenSecret = "kjOJW4YaYuq0esJP3Y0wmkEw51s4LFUyYkh2spYBbj8TO"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)


def percentage(part, whole):
    return 100 * float(part) / float(whole)

loc = "india"#sys.argv[1]     # location as argument variable
g = geocoder.osm(loc) # getting object that has location's latitude and longitude

closest_loc = api.trends_closest(g.lat, g.lng, lang="en")
trends = api.trends_place(closest_loc[0]['woeid'])

keyword = trends[0]["trends"][random.randint(1, len(trends[0]["trends"]))]["name"]#"#dogetothemoon"  # input("Please enter keyword or hashtag to search: ")
noOfTweet = 100  # int(input ("Please enter how many tweets to analyze: "))

tweets = tweepy.Cursor(api.search, q=keyword, lang="en", tweet_mode='extended').items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for tweet in tweets:
    tweet_list.append(tweet.full_text)
    analysis = TextBlob(tweet.full_text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.full_text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(tweet.full_text)
        negative += 1
    elif pos > neg:
        positive_list.append(tweet.full_text)
        positive += 1
    elif pos == neg:
        neutral_list.append(tweet.full_text)
        neutral += 1

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
print("total number: ", len(tweet_list))
print("positive number: ", len(positive_list))
print("negative number: ", len(negative_list))
print("neutral number: ", len(neutral_list))

labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]', 'Negative [' + str(negative) + '%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= " + keyword + "")
plt.axis('equal')
plt.show()

tweet_list.drop_duplicates(inplace=True)

# Cleaning Text (RT, Punctuation etc)
# Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]
# Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: ', " ", x)
rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
tw_list["text"] = tw_list.text.map(remove_rt)  # .map(rt)
tw_list["text"] = tw_list.text.str.lower()
tw_list.head(10)

# Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'compound'] = comp

print(tw_list.head(10))

# Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]


def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


# Count_values for sentiment
print(count_values_in_column(tw_list, "sentiment"))
tw_list.to_csv(keyword + ".csv", index=False)

# create data for Pie Chart
pc = count_values_in_column(tw_list,"sentiment")
names = pc.index
size = pc["Percentage"]

# Create a circle for the center of the plot
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(size, labels=names, colors=['green', 'blue', 'red'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()