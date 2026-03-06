from sklearn.preprocessing import MinMaxScaler


class MomentumScore:

    def compute(self, df):

        scaler = MinMaxScaler()

        features = df[["RSI","ROC","CCI"]].dropna()

        scaled = scaler.fit_transform(features)

        df.loc[features.index,"Momentum_Score"] = scaled.mean(axis=1)*100

        return df