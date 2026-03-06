import feedparser


class RSSParser:

    def __init__(self, url):
        self.url = url

    def parse(self):

        feed = feedparser.parse(self.url)

        news = []

        for entry in feed.entries:

            news.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })

        return news