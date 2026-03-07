import numpy as np
import pandas as pd


class RegimeAwareFeatures:

    TREND_MAP = {
        "Bearish": -1,
        "Sideways": 0,
        "Bullish": 1,
    }

    VOLATILITY_MAP = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    def add(self, df):

        df = df.copy()

        trend_series = df["Trend"] if "Trend" in df.columns else pd.Series("Sideways", index=df.index)
        volatility_series = (
            df["Volatility_Regime"]
            if "Volatility_Regime" in df.columns
            else pd.Series("Medium", index=df.index)
        )

        df["trend_regime_code"] = trend_series.astype(str).map(self.TREND_MAP).fillna(0)
        df["volatility_regime_code"] = volatility_series.astype(str).map(self.VOLATILITY_MAP).fillna(1)
        df["regime_adjusted_momentum"] = df["Momentum_Score"] * df["trend_regime_code"]
        df["regime_adjusted_return_5d"] = (
            df["return_5d"] / df["rolling_vol_20"].replace(0, np.nan)
        )
        df["regime_pressure_score"] = df["trend_regime_code"] - df["volatility_regime_code"]

        return df
