import json
from os import path

from hfer.server.app_config_provider import AppConfigProvider
from hfer.server.app_logic import AppLogic
from hfer.server.model_config_provider import ModelConfigProvider

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app_config = AppConfigProvider().app_config
print(app_config)
app.state.hfer = AppLogic(
    app_config["model_path"],
    app_config["image_input_dir"],
    app_config["json_output_dir"],
    ModelConfigProvider(app_config["config_path"]).config_data,
    app_config.get("bucket_name"),
)


# Define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


@app.get("/emo_from_img")
def getEmotionsFromImage(face_image_file):
    # print("Server.getEmotionsFromImage.name = " + face_image_file)
    json_str = app.state.hfer.get_face_emotions_from_file(face_image_file, 8, "text")
    # return HttpResponse("getEmotionsFromImage " + face_image_file)
    return json_str
