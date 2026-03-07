class TwitterCollector:

    def search(self, query, limit=100):

        try:
            import snscrape.modules.twitter as sntwitter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "snscrape is not installed. Install it with `pip install snscrape`."
            ) from exc

        tweets = []

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():

            if len(tweets) >= limit:
                break

            tweets.append({
                "date": tweet.date,
                "text": tweet.content
            })

        return tweets


if __name__ == "__main__":

    collector = TwitterCollector()

    tweets = collector.search("Reliance stock")

    print(tweets[:5])
