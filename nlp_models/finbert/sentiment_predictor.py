import torch
import torch.nn.functional as F


class FinBERTSentiment:

    def __init__(self, tokenizer, model):

        self.tokenizer = tokenizer
        self.model = model

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        )

        outputs = self.model(**inputs)

        scores = F.softmax(outputs.logits, dim=1)

        sentiment = scores.detach().numpy()

        return sentiment