/**
 * Some simple code written by William Hart (http://www.williamhart.info/) to test
 * the sentiment of a phrase.  An experiment in warning users off making abusive
 * comments on social media services such as twitter.
 */

(function (t, w, d) {
    "use strict";

    var split_words = function (phrase) {
        // http://stackoverflow.com/questions/18287323/split-and-count-words-in-string-node-js
        var ret = {},
            words = phrase
                .replace(/[.,?!;()"'\-]/g, " ")
                .replace(/\s+/g, " ")
                .toLowerCase()
                .split(" ");

        words.forEach(function (word) {
            if (!(ret.hasOwnProperty(word))) {
                ret[word] = 0;
            }
            ret[word] += 1;
        });

        return ret;
    },
        test_sentiment = function (phrase) {
            var words = split_words(phrase);
            console.log(words);
            return 3;
        };

    w.test_sentiment = test_sentiment;

})(this, window, document);
