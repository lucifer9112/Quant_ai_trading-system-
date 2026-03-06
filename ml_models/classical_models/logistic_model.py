from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LogisticModel:

    def train(self, df, target):

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        model = LogisticRegression()

        model.fit(X_train, y_train)

        return model