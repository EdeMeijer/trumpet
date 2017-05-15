import json
import os
import urllib.request as req

# Thanks to this guy who did the hard work of collecting all Trump tweets!
URI_FORMAT = 'http://www.trumptwitterarchive.com/data/realdonaldtrump/{}.json'
FIRST_YEAR = 2009
LAST_YEAR = 2017
CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'
print(CACHE_DIR)


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


def clean_html_entities(tweets):
    """
    Convert HTML entities to normal characters
    """
    # Todo
    return tweets


def get_yearly_tweets(year):
    """
    Get all original tweets from the given year as plain text, filtered and cleaned up
    """
    return clean_html_entities(extract_text(filter_oc(download_yearly_batch_cached(year))))


def get_all_tweets():
    """
    Get all original tweets as plain text, filtered and cleaned up
    """
    all_tweets = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        print('Downloading tweets from {}'.format(year))
        all_tweets.extend(get_yearly_tweets(year))
    return all_tweets


tweets_text = get_all_tweets()
print(len(tweets_text))
