from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinBERTLoader:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )