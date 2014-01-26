from tweepy.streaming import StreamListener
from tweepy import API as TweepyApi, OAuthHandler, Stream as TweepyStream

# import my settings from elsewhere to keep them secret
from local_settings import MY_SETTINGS


class StreamGatherer(StreamListener):
    """
    A utility class which gathers tweets from a stream and saves it
    """

    def __init__(self):
        super(StreamGatherer, self).__init__()
        self.results = []

    def on_data(self, data):
        self.results.append(data)
        return True

    def on_error(self, status):
        print "Received error code: {0}".format(status)


class GatherData(object):

    def __init__(self):
        """
        Builds an API connection and prepares to download data
        """
        self.auth = OAuthHandler(
            MY_SETTINGS.consumer_key,
            MY_SETTINGS.consumer_secret
        )
        self.auth.set_access_token(
            MY_SETTINGS.access_token_key,
            MY_SETTINGS.access_token_secret
        )

        self.api = TweepyApi(self.auth)

        self.good = []
        self.bad = []

    def search_tweets(self, search, count=15):
        """
        Gets a list of 15 tweets with the given search terms

        :param search: the string to search for in the tweets
        """
        result = self.api.search(q=search, count=count, lang='en')
        return [x.text.encode('ascii', 'ignore').replace('\n', '') for x in result]

    def get_recent_tweets(self, count=15):
        """
        Returns the most recent tweets

        :param count: the number of tweets to return
        """

        listener = StreamGatherer()
        stream = TweepyStream(self.auth, listener)
        stream.sample(count)

        while stream.running:
            # wait until we are finished
            pass

        return listener.results

    def gather_tweets(self, num_good=100, num_bad=150):
        """
        Queries the twitter API for a number of "good and bad" tweets,
        """

        # hackish "random" method because stream.sample() 401s like a boss
        self.good = self.search_tweets("emberjs", 200)
        self.good += self.search_tweets("nba", 200)
        self.good += self.search_tweets("superbowl", 200)
        self.good += self.search_tweets("science", 200)
        self.good += self.search_tweets("bieber", 200)
        self.good += self.search_tweets("NASA", 200)
        self.good += self.search_tweets("forest", 200)
        #self.good = self.get_recent_tweets(num_good)

        self.bad = self.search_tweets("star wars", 200)
        self.bad += self.search_tweets("#starwars", 200)

    def write_tweets(self, path):
        """
        Writes the gathered tweets to a file in the preferred format

        ;param path: the path to the file to write tweets to
        """
        with open(path, 'w') as f:
            for t in self.good:
                f.write(t + "0\n")

            for t in self.bad:
                f.write(t + "1\n")
