from sklearn.feature_selection import SelectKBest, f_classif


class FeatureSelector:

    def select(self, df, target, k=10):

        X = df.drop(columns=[target])
        y = df[target]

        selector = SelectKBest(score_func=f_classif, k=k)

        X_new = selector.fit_transform(X, y)

        selected = X.columns[selector.get_support()]

        return selected