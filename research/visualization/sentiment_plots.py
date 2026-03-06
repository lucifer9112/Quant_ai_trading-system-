import matplotlib.pyplot as plt


class SentimentPlots:

    def plot_sentiment(self, df):

        if "sentiment" not in df.columns:
            return

        plt.figure(figsize=(12,5))

        plt.plot(df["Date"], df["sentiment"])

        plt.title("News Sentiment Trend")

        plt.show()