from clean_tweet import TweetClassifier as TC


def run_test(val):
    print "{0} >> {1}".format(t.predict(val), val)


t = TC("train_data.txt")
t.train()

run_test("Another test")
run_test("A horse test")
run_test("A horsey test")
run_test("Equestrian")
run_test("Test horse eat horse hungry horse")

print "_____________________________________"

print t.vocab

print "_____________________________________"

print t.probabilities
