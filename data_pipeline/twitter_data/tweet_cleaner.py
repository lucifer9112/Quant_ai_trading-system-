import re


class TweetCleaner:

    def clean(self, text):

        text = text.lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"@\w+", "", text)

        text = re.sub(r"#", "", text)

        text = re.sub(r"[^a-zA-Z ]", "", text)

        return text.strip()