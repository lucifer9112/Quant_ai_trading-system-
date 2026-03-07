class NewsSentiment:

    def score(self, text):

        if text is None:
            return 0

        try:
            from textblob import TextBlob
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "textblob is not installed. Install it with `pip install textblob`."
            ) from exc

        return TextBlob(text).sentiment.polarity
