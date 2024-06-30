import json


class ModelConfigProvider:
    def __init__(self):
        self._config_data = {}

    @property
    def config_data(self):
        return self._config_data
