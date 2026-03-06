from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class VolumeIndicators:

    def add(self, df):

        obv = OnBalanceVolumeIndicator(
            df["Close"],
            df["Volume"]
        )

        df["OBV"] = obv.on_balance_volume()

        vwap = VolumeWeightedAveragePrice(
            df["High"],
            df["Low"],
            df["Close"],
            df["Volume"]
        )

        df["VWAP"] = vwap.volume_weighted_average_price()

        df["Volume_MA20"] = df["Volume"].rolling(20).mean()

        return df