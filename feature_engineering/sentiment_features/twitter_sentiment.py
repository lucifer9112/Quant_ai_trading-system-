from textblob import TextBlob


class TwitterSentiment:

    def score(self, text):

        return TextBlob(text).sentiment.polarity