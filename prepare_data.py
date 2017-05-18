"""
This script contains functionality for downloading, cleaning up and converting Donald Trump tweets to a numpy data
format suitable for training a character level modelling network.
"""
import html
import json
import os
import random
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
    Given a year, download the JSON encoded batch of Trump tweets for that year and returns the JSON string
    """
    print('Downloading tweets from {}'.format(year))
    with req.urlopen(URI_FORMAT.format(year)) as uri:
        return uri.read().decode()


def download_yearly_batch_cached(year):
    """
    Given a year, fetch the JSON encoded Trump tweets from cache or download it and then cache it. Returns the
    parsed JSON.
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
    return [tweet for tweet in tweets if is_oc(tweet)]


def is_oc(tweet):
    """
    Check if a tweet is original content and not a retweet or reply
    """
    if tweet['is_retweet']:
        return False
    if tweet['in_reply_to_user_id_str'] is not None:
        return False
    if '@realDonaldTrump' in tweet['text']:
        # Here he's copying other peoples tweets and responding to them, but they're not replies or retweets
        return False
    return True


def extract_tweet_text(tweets):
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
    return cleanup(extract_tweet_text(filter_oc(download_yearly_batch_cached(year))))


def get_all_tweets():
    """
    Get all original tweets as plain text, filtered and cleaned up
    """
    all_tweets = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        all_tweets.extend(get_yearly_tweets(year))
    return all_tweets


def count_chars(tweets):
    """
    Count the occurrence of all characters in the given tweets. Returns a dictionary with characters as keys and
    the integer number of occurrences as values.
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
    Get a list of characters that have too few occurrences and should be excludes from the data set
    """
    return [char for char, num in count_chars(tweets).items() if num < MIN_CHAR_OCCURRENCES]


def exclude_tweets_with_rare_chars(tweets):
    """
    Exclude tweets that contain characters with too little overall occurrences
    """
    excludes = get_char_exclude_list(tweets)
    return [tweet for tweet in tweets if not any(char in tweet for char in excludes)]


def get_features(tweet, unique_chars):
    """
    Given a tweet and a character list, determine the 0-based integer class for every character in the tweet and return
    the list of classes. Will prepend a special class with index len(unique_chars) to the list, which indicates the
    start of the tweet. This allows the model to learn to predict the first character from scratch.
    """
    return [len(unique_chars)] + [unique_chars.index(char) for char in tweet]


def get_labels(tweet, unique_chars):
    """
    Given a tweet and a character list, determine the 0-based integer class for every character in the tweet and return
    the list of classes. Will append a special class with index len(unique_chars) to the list, which indicates the
    end of the tweet. This allows the model to learn to predict when the tweet is done.
    """
    return [unique_chars.index(char) for char in tweet] + [len(unique_chars)]


def get_unique_chars(tweets):
    """
    Returns a list of unique characters occurring in the given tweets, sorted by natural order
    """
    return sorted(char for char, _ in count_chars(tweets).items())


def create_training_data():
    """
    Create all data required for training. Will collect all tweets and transform it to trainable features and labels.

    Returns:
      features: 3D numpy array of shape [num_tweets, max_time_steps, 1] where max_time_steps is the number of characters
        of the longest tweet in the data set + 1, to accommodate the 'start of tweet' special character followed by
        the indices of the characters in every tweet. Zero padded to max_time_steps for shorter tweets.
      labels: 3D numpy array with same shape as `features`. Contains the indices of the characters in every tweet,
        followed by a special label with class len(unique_chars) that indicates the end of the tweet. Zero padded to
        max_time_steps for shorter tweets.
      mask: 2D numpy array of shape [num_tweets, max_time_steps]. Contains 1's for time steps that contain actual
        feature/label pairs, and 0's for the zero-padded steps of shorter tweets. Needed to ignore the training error
        on padded time steps.
      settings: dictionary that contains the unique characters used for the training data, and the maximum number of
        time steps. Needed for training and being able to reproduce characters from integer classes for sampling
        synthetic tweets.
    """
    # Collect all usable tweets and shuffle them deterministically (shuffling is important for training)
    all_tweets = exclude_tweets_with_rare_chars(get_all_tweets())
    random.seed(12345)
    random.shuffle(all_tweets)

    print("got all {} tweets, creating features and labels".format(len(all_tweets)))

    unique_chars = get_unique_chars(all_tweets)

    # The maximum number of time steps is the longest tweet length + 1 for the special 'start tweet' character.
    max_steps = max(len(tweet) + 1 for tweet in all_tweets)

    # Create the numpy array for all features and labels
    features = np.zeros([len(all_tweets), max_steps], dtype=int)
    labels = np.zeros_like(features)
    mask = np.zeros([len(all_tweets), max_steps], dtype=float)

    for i in range(len(all_tweets)):
        tweet = all_tweets[i]
        num_steps = len(tweet) + 1

        features[i, :num_steps] = get_features(tweet, unique_chars)
        labels[i, :num_steps] = get_labels(tweet, unique_chars)
        mask[i, :num_steps] = 1

    return features, labels, mask, {'chars': unique_chars, 'maxSteps': max_steps}


def export_training_data():
    """
    Export features, labels, mask and settings to files so that it can be used by the training script
    """
    features, labels, mask, settings = create_training_data()

    np.save(CACHE_DIR + '/features.npy', features)
    np.save(CACHE_DIR + '/mask.npy', mask)
    np.save(CACHE_DIR + '/labels.npy', labels)
    with open(CACHE_DIR + '/settings.json', 'w') as file:
        json.dump(settings, file)


if __name__ == "__main__":
    export_training_data()
