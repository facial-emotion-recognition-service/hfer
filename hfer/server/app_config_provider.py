import os


class AppConfigProvider:
    def __init__(self):
        self._model_path = os.environ.get("MODEL_PATH")
        self._config_path = os.environ.get("CONFIG_PATH")
        self._image_input_dir = os.environ.get("IMAGE_INPUT_DIR")
        self._json_output_dir = os.environ.get("JSON_OUTPUT_DIR")
        self._bucket_name = os.environ.get("BUCKET_NAME", None)
        # Note: Only set BUCKET_NAME if loading the model from GCS

    @property
    def app_config(self):
        result = {
            "model_path": self._model_path,
            "config_path": self._config_path,
            "image_input_dir": self._image_input_dir,
            "json_output_dir": self._json_output_dir,
            "bucket_name": self._bucket_name,
        }

        return result
