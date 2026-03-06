from textblob import TextBlob


class AspectSentiment:

    def analyze(self, text):

        blob = TextBlob(text)

        return blob.sentiment.polarity