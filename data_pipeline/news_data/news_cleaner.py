import re


class NewsCleaner:

    def clean_text(self, text):

        text = text.lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"[^a-zA-Z ]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def clean_news(self, news_list):

        cleaned = []

        for item in news_list:

            cleaned.append({
                "title": self.clean_text(item["title"]),
                "link": item["link"],
                "published": item["published"]
            })

        return cleaned
