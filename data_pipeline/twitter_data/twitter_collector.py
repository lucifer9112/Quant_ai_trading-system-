class TwitterCollector:

    @staticmethod
    def _twitter_module():

        try:
            import snscrape.modules.twitter as sntwitter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "snscrape is not installed. Install it with `pip install snscrape`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "snscrape could not be imported. This often means the installed version "
                "is incompatible with the current Python runtime."
            ) from exc

        return sntwitter

    def search(self, query, limit=100):

        tweets = []

        try:
            scraper = self._twitter_module().TwitterSearchScraper(query)

            for tweet in scraper.get_items():

                if len(tweets) >= limit:
                    break

                tweets.append({
                    "date": tweet.date,
                    "text": tweet.content
                })
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Twitter collection failed for query '{query}': {exc}"
            ) from exc

        return tweets


if __name__ == "__main__":

    collector = TwitterCollector()

    tweets = collector.search("Reliance stock")

    print(tweets[:5])
