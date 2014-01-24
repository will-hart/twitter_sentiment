from clean_tweet import TweetClassifier as TC
t = TC("train_data.txt")
t.train()

print t.predict("A test")
print t.predict("Another test")
print t.predict("A horsey test")
print t.predict("Equestrian")
print t.predict("Test horse eat horse hungry horse")
