import os


class AppConfigProvider:
    def __init__(self):
        self._model_path = os.environ.get("MODEL_PATH")
        self._bucket_name = os.environ.get("BUCKET_NAME", None)
        # Note: Only set BUCKET_NAME if loading the model from GCS

    @property
    def app_config(self):
        result = {
            "model_path": self._model_path,
            "bucket_name": self._bucket_name,
        }

        return result
