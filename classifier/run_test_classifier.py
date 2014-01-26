import os
import shutil
import time

from clean_tweet import TweetClassifier as TC
from gather_data import GatherData


def run_test(val, expected):
    print "{0} (exp {1}) >> {2}".format(t.predict(val), expected, val)

# Start by gathering some data.
g = GatherData()
"""
# If we have an existing training set, this becomes the new test set (just for variety)
if os.path.isfile("train_data.txt"):
    print "Old training data is the new test data..."
    shutil.copyfile("train_data.txt", "test_data.txt")
else:
    print "Gathering new test data"
    g.gather_tweets()
    g.write_tweets("test_data.txt")
    time.sleep(3)

# gather new training data
print "Gathering training data..."
g.gather_tweets()
g.write_tweets("train_data.txt")
"""

# train the classifier
print "Training the classifier..."
t = TC("train_data.txt")
t.train()
print t.probabilities.shape

# test the classifier
print "Testing the classifier..."
tested = 0
correct = 0

with open('test_data.txt', 'r') as f:
    for line in f.readlines():
        tested += 1
        line = line[:-1]
        if t.predict(line[:-1]) == int(line[-1]):
            correct += 1

print "Tested {0} tweets, got {1} correct ({2:.0%})".format(tested, correct, correct/float(tested))
