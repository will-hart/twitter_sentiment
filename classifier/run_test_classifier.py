from tweet_classifier import TweetClassifier as TC


def run_test(val, expected):
    print "{0} (exp {1}) >> {2}".format(t.classify(val), expected, val)

# create the classifier
t = TC()

# Start by gathering some data.
print "Gathering data..."
t.fetch_data()

# train the classifier
print "Training the classifier..."
t.train("train_data.txt")

# test the classifier
print "Testing the classifier..."
tested = 0
correct = 0

with open('test_data.txt', 'r') as f:
    for line in f.readlines():
        tested += 1
        line = line[:-1]
        if t.classify(line[:-1]) == int(line[-1]):
            correct += 1

print "Tested {0} tweets, got {1} correct ({2:.0%})".format(tested, correct, correct/float(tested))
