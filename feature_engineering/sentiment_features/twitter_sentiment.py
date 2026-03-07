class TwitterSentiment:

    def score(self, text):

        try:
            from textblob import TextBlob
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "textblob is not installed. Install it with `pip install textblob`."
            ) from exc

        return TextBlob(text).sentiment.polarity
