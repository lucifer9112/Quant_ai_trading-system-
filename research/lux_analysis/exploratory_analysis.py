import pandas as pd


class ExploratoryAnalysis:

    def summary(self, df):

        print("Dataset shape:", df.shape)

        print("\nColumns:")
        print(df.columns)

        print("\nStatistics:")
        print(df.describe())

        print("\nMissing values:")
        print(df.isna().sum())