from textblob import TextBlob


class NewsSentiment:

    def score(self, text):

        if text is None:
            return 0

        return TextBlob(text).sentiment.polarity