import os
import json


class ConfigProvider:
    def __init__(self):
        self._model_path = os.environ.get("MODEL_PATH")
        self._bucket_name = os.environ.get("BUCKET_NAME", None)
        self._labels = {}
        with open(os.environ.get("CONFIG_PATH"), "r") as json_config_file:
            self._labels = json.load(json_config_file)
        # Note: Only set BUCKET_NAME if loading the model from GCS

    @property
    def config(self):
        result = {
            "model_path": self._model_path,
            "labels": self._labels,
            "bucket_name": self._bucket_name,
        }

        return result
