import yaml


class ConfigLoader:

    def __init__(self, path):

        self.path = path

    def load(self):

        with open(self.path, "r") as f:

            return yaml.safe_load(f)