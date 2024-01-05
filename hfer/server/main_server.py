import json
from os import path

from fastapi import FastAPI

from hfer.server.app_config_provider import AppConfigProvider
from hfer.server.app_logic import AppLogic
from hfer.server.model_config_provider import ModelConfigProvider

# settings.configure(
#     DEBUG=True,
#     SECRET_KEY="4l0ngs3cr3tstr1ngw3lln0ts0l0ngw41tn0w1tsl0ng3n0ugh",
#     IGNORABLE_404_URLS=[r"^favicon\.ico$"],
#     ROOT_URLCONF=sys.modules[__name__],
# )

app = FastAPI()


# Define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


@app.get("/emo_from_img")
def getEmotionsFromImage(face_image_file):
    # print("Server.getEmotionsFromImage.name = " + face_image_file)
    hferapp.get_face_emotions_from_file(face_image_file, 8, "text")
    # return HttpResponse("getEmotionsFromImage " + face_image_file)
    return json.load(path.join(json_output_dir, face_image_file))


if __name__ == "__main__":
    appConfigProvider = AppConfigProvider()
    app_config = appConfigProvider.app_config
    model_config_path = app_config["config_path"]
    modelConfigProvider = ModelConfigProvider(model_config_path)

    config_path = app_config["config_path"]
    model_path = app_config["model_path"]
    image_input_dir = app_config["image_input_dir"]
    json_output_dir = app_config["json_output_dir"]
    bucket_name = app_config.get("bucket_name")

    model_config = modelConfigProvider.config_data

    hferapp = AppLogic(
        model_path, image_input_dir, json_output_dir, model_config, bucket_name
    )
