"""
Use Naive Bayesian classification techniques to determine if a tweet is "abusive" (in this case talking about horses)
"""
import numpy as np
import os
import shutil
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import time

from gather_data import GatherData


class TweetClassifier(object):
    """
    Classifies tweets into good or bad depending on their horse quota.  Usage:

        from clean_tweet import TweetClassifier as TC
        t = TC("test_data.txt")
        t.train()
    """

    def __init__(self, file_path=None):
        self.path = file_path
        self.results = []
        self.tweets = []
        self.vocab = {}
        self.probabilities = []
        self.vectoriser = None
        self.transformer = None
        self.classifier = None

    def fetch_data(self):
        """
        Fetches test/training data from twitter directly
        """
        g = GatherData()

        # If we have an existing training set, this becomes the new test set (just for variety)
        if os.path.isfile("train_data.txt"):
            shutil.copyfile("train_data.txt", "test_data.txt")
        else:
            g.gather_tweets()
            g.write_tweets("test_data.txt")
            time.sleep(3)

        # gather new training data
        g.gather_tweets()
        g.write_tweets("train_data.txt")

    def load_dataset(self, path=None):
        """
        Loads a data set from a text file.  The data set should have one tweet per line, with the last character
        being a `0` for "good" or `1` for bad.  Each tweet should start on a new line.  For instance:

            A horse is a horse of course1
            Jack and the beanstalk0

        :param path: The path to the text file that results are kept in (or None if the default should be used)
        """
        with open(path if path is not None else self.path) as f:
            line_count = 0
            for line in f:
                line_count += 1
                if len(line) > 2:
                    if line[-1] == '\n':
                        line = line[:-1] # remove trailing newline

                    # check the line is in the correct format
                    try:
                        id = int(line[-1])
                    except ValueError:
                        print "Error on line {0} - does not end in a number".format(line_count)
                        print "   {0}".format(line)
                    else:
                        self.results.append(int(line[-1]))
                        yield line[:-1]

    def train(self, path=None, dump_inputs=False):
        """
        Teaches the classifier based on the data set passed in the constructor

        :param path: The file path for training data, or None if one provided in the constructor
        :param dump_inputs: Set to True if you want the input file to be written to input.txt in svmlight format
        """

        # check for a path
        if path is None and self.path is None:
            raise ValueError("No path has been given for training data - unable to train")
        elif path is not None:
            self.path = path

        # vectorise the text
        self.vectoriser = CountVectorizer(lowercase=True, strip_accents='unicode')
        res = self.vectoriser.fit_transform(self.load_dataset(self.path))

        # Apply "Term Frequency times Inverse Document Frequency" methodology.  The term frequency part is
        # not really required given the fixed tweet length but very common words should be ignored so inverse document
        # frequency is relevant
        self.transformer = TfidfTransformer()
        res = self.transformer.fit_transform(res)

        # create a numpy array of expected results
        expected = np.asarray(self.results)

        # write the output to SVM if desired
        if dump_inputs:
            dump_svmlight_file(res, expected, f="input.txt")

        self.vocab = self.vectoriser.get_feature_names()

        # use a multinomial classifier to generate probabilities
        self.classifier = MultinomialNB()
        self.classifier.fit(res, expected)
        self.probabilities = self.classifier.feature_log_prob_

    def classify(self, tweet):
        """
        Predicts whether a tweet is horsey using the trained classifier
        """
        tweets = [tweet]
        tweet_vec = self.vectoriser.transform([tweet])
        tweet_tf = self.transformer.transform(tweet_vec)
        result = self.classifier.predict(tweet_tf)[0]
        return result
