import time

from clean_tweet import TweetClassifier as TC
from gather_data import GatherData


def run_test(val, expected):
    print "{0} (exp {1}) >> {2}".format(t.predict(val), expected, val)

# Start by gathering some data
g = GatherData()
g.gather_tweets()
g.write_tweets("train_data.txt")

time.sleep(3)
g.gather_tweets()
g.write_tweets("test_data.txt")

# train the classifier
t = TC("train_data.txt")
t.train()

# test the classifier
with open('test_data.txt', 'r') as f:
    for line in f.readlines():
        run_test(line[:-1], line[-1])
