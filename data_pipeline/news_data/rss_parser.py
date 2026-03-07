class RSSParser:

    def __init__(self, url):
        self.url = url

    def parse(self):

        try:
            import feedparser
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "feedparser is not installed. Install it with `pip install feedparser`."
            ) from exc

        feed = feedparser.parse(self.url)

        news = []

        for entry in feed.entries:

            news.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })

        return news
