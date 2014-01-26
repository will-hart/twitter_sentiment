from flask import Flask, request
from flask_cors import cross_origin
import os

from tweet_classifier import TweetClassifier

print ""
print ""
print "----------------------------------------"
print "|                                      |"
print "|     RUNNING TWEET CLASSIFIER APP     |"
print "|                                      |"
print "----------------------------------------"
print ""
print ""
print "Initialising application"
app =  Flask(__name__)
app.debug = True

print "Initialising Tweet classifier..."
classifier = TweetClassifier()

if os.path.isfile("train_data.txt"):
    print " > using cached training data"
else:
    print " > fetching training data"
    classifier.fetch_data()

print " > training classifier"
classifier.train("train_data.txt")

@app.route("/", methods=["GET"])
@cross_origin()
def validate_tweet():
    tweet = request.args.get('tweet')
    return str(classifier.classify(tweet))

print "Starting app"
app.run()

print ""
print "----------------------------------------"
print ""
