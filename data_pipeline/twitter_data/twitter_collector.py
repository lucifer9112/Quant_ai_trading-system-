import snscrape.modules.twitter as sntwitter


class TwitterCollector:

    def search(self, query, limit=100):

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