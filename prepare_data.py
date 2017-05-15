import html
import json
import os
import urllib.request as req

import numpy as np
from unidecode import unidecode

# Thanks to this guy who did the hard work of collecting all Trump tweets!
URI_FORMAT = 'http://www.trumptwitterarchive.com/data/realdonaldtrump/{}.json'
FIRST_YEAR = 2009
LAST_YEAR = 2017
CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'
MIN_CHAR_OCCURRENCES = 500


def download_yearly_batch(year):
    """
    Given a year, download the JSON encoded batch of Trump tweets for that year
    """
    with req.urlopen(URI_FORMAT.format(year)) as uri:
        return uri.read().decode()


def download_yearly_batch_cached(year):
    """
    Given a year, fetches the JSON encoded and parsed batch of Trump tweets for that year from cache or downloads
    """
    path = '{}/{}.json'.format(CACHE_DIR, year)
    if not os.path.exists(path):
        with open(path, 'w') as file:
            file.write(download_yearly_batch(year))

    with open(path, 'r') as file:
        return json.load(file)


def filter_oc(tweets):
    """
    Filter out retweets and replies, because we are only interested in original Trump prose
    """
    return [tweet for tweet in tweets if not tweet['is_retweet'] and tweet['in_reply_to_user_id_str'] is None]


def extract_text(tweets):
    """
    Just grab 'em by the "text" fields
    """
    return [tweet['text'] for tweet in tweets]


def cleanup(tweets):
    """
    Convert HTML entities to normal characters and convert to ASCII
    """
    return [unidecode(html.unescape(tweet)) for tweet in tweets]


def get_yearly_tweets(year):
    """
    Get all original tweets from the given year as plain text, filtered and cleaned up
    """
    return cleanup(extract_text(filter_oc(download_yearly_batch_cached(year))))


def get_all_tweets():
    """
    Get all original tweets as plain text, filtered and cleaned up
    """
    all_tweets = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        print('Downloading tweets from {}'.format(year))
        all_tweets.extend(get_yearly_tweets(year))
    return all_tweets


def count_chars(tweets):
    """
    Count the occurrence of all characters in the given tweets
    """
    counts = {}
    for tweet in tweets:
        for char in tweet:
            if char not in counts:
                counts[char] = 0
            counts[char] += 1
    return counts


def get_char_exclude_list(tweets):
    """
    Get a list of characters that have too little occurrences and should be excludes from the data set
    """
    return [char for char, num in count_chars(tweets).items() if num < MIN_CHAR_OCCURRENCES]


def exclude_tweets_with_rare_chars(tweets):
    """
    Exclude tweets that contain characters with too little overall occurrences
    """
    excludes = get_char_exclude_list(tweets)
    return [tweet for tweet in tweets if not any(char in tweet for char in excludes)]


def get_char_feature(tweet, chars):
    """
    Given a tweet and a character list, determine the 0-based integer class for every character in the tweet and return
    the list of classes.
    """
    return [chars.index(char) for char in tweet.lower()]


def get_capitalization_feature(tweet):
    """
    Given a tweet, create a binary feature of whether a character is capitalized. Crucial for genuine Trump tweets.
    """
    return [0 if char == char.lower() else 1 for char in tweet]


all_tweets = exclude_tweets_with_rare_chars(get_all_tweets())
print("got all tweets, creating features and labels")

# Unique chars in lower case. We will add an extra binary input for capitalization.
unique_chars = sorted(char for char, _ in count_chars([t.lower() for t in all_tweets]).items())

# Create the numpy array for all features and labels
# There are 2 features in total. Dimensions are [num_examples x max_time_steps x num_features]
features = np.zeros([len(all_tweets), max(len(tweet) + 1 for tweet in all_tweets), 2], dtype=int)
lengths = np.zeros([len(all_tweets), 1], dtype=int)
labels = np.zeros_like(features)

for i in range(len(all_tweets)):
    tweet = all_tweets[i]
    num_steps = len(tweet)

    char_feature = get_char_feature(tweet, unique_chars)
    capitalization_feature = get_capitalization_feature(tweet)

    features[i, :num_steps, 0] = char_feature
    features[i, :num_steps, 1] = capitalization_feature
    lengths[i, 0] = num_steps

    labels[i, :num_steps - 1, 0] = char_feature[1:]
    labels[i, num_steps - 1, 0] = len(unique_chars)
    labels[i, :num_steps - 1, 1] = capitalization_feature[1:]

np.save(CACHE_DIR + '/features.npy', features)
np.save(CACHE_DIR + '/lengths.npy', lengths)
np.save(CACHE_DIR + '/labels.npy', labels)

# Save the list of unique characters
with open(CACHE_DIR + '/settings.json', 'w') as file:
    json.dump({'chars': unique_chars}, file)