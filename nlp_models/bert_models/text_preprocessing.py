import re


class TextPreprocessor:

    def clean(self, text):

        text = text.lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"[^a-zA-Z ]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()