import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class XGBoostModel:

    def train(self, df, target):

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        model = xgb.XGBClassifier()

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print("XGBoost Accuracy:", acc)

        return model