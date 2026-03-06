import pandas as pd
import numpy as np


class VolatilityRegime:

    def classify(self, df):

        vol = df["Historical_Vol"]

        df["Volatility_Regime"] = pd.cut(
            vol,
            bins=[
                -np.inf,
                vol.quantile(0.33),
                vol.quantile(0.66),
                np.inf
            ],
            labels=["Low","Medium","High"]
        )

        return df