class FeatureRegistry:

    def __init__(self):

        self._transforms = {}

    def register(self, name, transform):

        self._transforms[name] = transform
        return self

    def names(self):

        return list(self._transforms.keys())

    def apply(self, df, selected=None):

        names = selected or self.names()

        for name in names:
            df = self._transforms[name](df)

        return df
