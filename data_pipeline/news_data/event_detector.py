class EventDetector:

    EVENTS = [
        "budget",
        "rbi",
        "earnings",
        "interest rate",
        "policy",
        "inflation"
    ]

    def detect(self, text):

        for event in self.EVENTS:

            if event in text.lower():
                return event

        return None