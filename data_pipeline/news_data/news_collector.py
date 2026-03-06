try:
    from .rss_parser import RSSParser
except ImportError:
    from data_pipeline.news_data.rss_parser import RSSParser


class NewsCollector:

    def __init__(self):

        self.sources = [
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.moneycontrol.com/rss/business.xml"
        ]

    def collect(self):

        all_news = []

        for source in self.sources:

            parser = RSSParser(source)

            data = parser.parse()

            all_news.extend(data)

        return all_news


if __name__ == "__main__":

    collector = NewsCollector()

    news = collector.collect()

    print(news[:5])
