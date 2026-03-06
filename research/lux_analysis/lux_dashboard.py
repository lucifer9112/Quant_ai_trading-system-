import lux
import pandas as pd


class LuxDashboard:

    def analyze(self, df):

        lux.config.default_display = "lux"

        print("Lux automatic analysis enabled")

        return df