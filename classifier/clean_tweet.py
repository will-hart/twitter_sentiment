"""
Use Naive Bayesian classification techniques to determine if a tweet is "abusive" (in this case talking about horses)
"""

import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


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
        self.classifier = None
        self.probabilities = []

    def load_dataset(self, path=None):
        """
        Loads a data set from a text file.  The data set should have one tweet per line, with the last character
        being a `0` for "good" or `1` for bad.  Each tweet should start on a new line.  For instance:

            A horse is a horse of course1
            Jack and the beanstalk0

        :param path: The path to the text file that results are kept in (or None if the default should be used)
        """
        with open(path if path is not None else self.path) as f:
            for line in f:
                line = line[:-1] # remove trailing newline
                self.results.append(int(line[-1]))
                yield line[:-1]

    def train(self, dump_inputs=False):
        """
        Teaches the classifier based on the data set passed in the constructor

        :param dump_inputs: Set to True if you want the input file to be written to input.txt in svmlight format
        """

        # vectorise the text
        self.vectoriser = CountVectorizer(lowercase=True, strip_accents='unicode')
        res = self.vectoriser.fit_transform(self.load_dataset())

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

    def predict(self, tweet):
        """
        Predicts whether a tweet is horsey using the trained classifier
        """
        tweets = [tweet]
        tweet_vec = self.vectoriser.transform([tweet])
        tweet_tf = self.transformer.transform(tweet_vec)
        return self.classifier.predict(tweet_tf)[0]
