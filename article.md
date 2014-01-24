# Cleaning up Twitter

I follow one particular person on Twitter, originally because they tweeted a couple of funny but random things, which
made me `lol`.  In the first few days I thought they were one of those "joke" accounts, which only post funny things, yet
pretty soon afterwards they tweeted something that horrified me and I realised this person was sorta kind undercover.

You see this person had been a victim of some pretty horrible harassment on Twitter, which included death threats and
very graphic threats of rape.  Whilst unfortunately this kind of unsocial behaviour is part and parcel of the anonymity
the internet allows (google the Greater Internet F&$kwad Theory), I was more horrified to read a follow up article in
which it was shown how little the police and Twitter themselves were prepared to do to prevent this kind of despicable
behaviour.  In many ways though, it is unclear what the authorities can do - the abuse is very widespread and although
the abusers likely leave IP evidence, due to privacy laws it is not necessarily easy to follow these leads up.  Two days
later I was happy to see that the UK government had jailed two people (a man and a woman) who had taken part in this
kind of online abuse.

How much should Twitter and other social media outlets be required to police what goes using their services? Certainly
in the past phone companies were not held to account for what people said over the phone lines, but in a world where
tweets and the like are public and persistent, is there a new burden of responsibility for these companies? And if there
is, what on earth can they do about it?

I don't know how much demographic or psychological analysis has been done on this sort of behaviour, but from some wild
supposition I would imagine that you can divide the perpetrators into two different groups.  The first is those who are
doing it for "a bit of a laugh" or "all in good fun", because they get their kicks from it - without considering the impact.
The second group its potentially a symptom or a consequence of some form of mental illness.  The behaviour of the first
groups can likely be corrected.  Whilst no less terrifying, the second category is unlikely to be managed through actions
undertaken by Twitter.

## What can be done?

One possible way to "jolt" the first group into modifying their behaviour is through a visual cue in the browser, which
alerts them if the tweet they have typed (and are about to "send") appears to be abusive.  For instance, the message
could read:

> WARNING - the message you have typed could be abusive, note that online abuse can be a criminal offence.

If coupled with flashing red lights and warning images, the casual abuser could potentially be turned off writing their
abusive tweet, particularly if they are informed that their IP address has been logged.

This kind of task is relatively common, and is a kind of "classification problem", where a computer must categories a
data point, usually based on a small and finite number of possible states.  In the case of natural language (i.e. text),
this technique is frequently known as "sentiment analysis", which refers to an algorithm which looks over a sentence or
slab of text, and tries to work out if the *mood* of the text is positive or negative based on the prevalence of certain
words or word patterns.

## Sentiment analysis

The basic approach is often quite simple:

1. Work out the frequency of words in the section of text
2. Apply some simple rules to these words frequencies to detect if the text should be considered good or bad

The first part is just a string manipulation technique, and particularly for something like Tweets (with 160 characters)
would be quite easy to do.  One complication may be that the use of abbreviations and "text speak" (or whatever the kids
are calling it these days) would mean that the number of words that would need to be tracked would grow.

To determine the rules required in step 2, a number of techniques exist.  These usually take a data set, such as a whole
list of tweets, with each data point having its classification.  A certain proportion of the data points are taken to
"teach" the algorithm, whilst others are used to test the algorithm once it has finsihed learning.

## Classification

I decided to use a Naive Bayes approach, which uses probability theory to determine whether (based on word frequency)
a tweet is good or bad.  For instance what we are trying to do is answer the following question:

> What is the probability the tweet is bad *given* it has the following words in it: ....

In mathematical symbols, this could look something like the following (from the `scikit-learn` documentation):

\[P(y|x_1, ..., x_n) = \frac{P(y)\Pi_{i=1}^nP(x_i|y)}{P(x_1, ..., x_n)}\]

Where `y` is the "good"/"bad" classification and `x` variables are the words in the tweet.

Training a classifier is relatively simple with a good dataset and something like Python's `scikit-learn`.  This can be
installed by typing:

    $ pip install scipy
    $ pip install scikit-learn

However on Windows this can sometimes be a bit difficult so I just use [WinPython](http://winpython.sourceforge.net/)
for these kinds of tasks.

### The learning data set

To teach our classification algorithm we need some kind of learning data set. This would contain as many tweets as we
could find and a flag to indicate which of these is considered "bad".  As I don't really want to upload and work with a
data set filled with horrible words and phrases, lets construct a fictional example, where we are trying to detect if
our tweets are related to horses. For instance the following (made up) tweets are horse related:

> I'm going to gallop around the fields
> I'm going to the horse races
> My mare is very cute
> I just got myself a new saddle

Whilst the following would not be horse related

> My coffee tastes like bilge water
> It's raining cats and dogs
> I was just horsing around
> I am as hungry as a horse

This system may run into difficulties, because the word "horse" can be used in a number of non-equine contexts, but this
is part of the challenge of natural language processing.  The larger the dataset, the more likely it is that the algorithm
will be able to detect these similar, but not "bad" messages.


