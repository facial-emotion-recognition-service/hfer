import json
from os import makedirs, path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from hfer.server.app_config_provider import AppConfigProvider
from hfer.server.app_logic import AppLogic
from hfer.server.model_config_provider import ModelConfigProvider

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


@app.post("/upload_image")
def uploadImage(image: UploadFile):
    ## Extract faces from image
    image = app.state.hfer.convert_upload_to_array(image)
    face_ids = app.state.hfer.get_faces_from_image(image)

    ## Get annotated image
    if face_ids:
        annotated_image, colors = app.state.hfer.get_annotated_image(
            image, face_ids
        )
        annotated_image = app.state.hfer.convert_array_to_base64(
            annotated_image
        )
    else:
        annotated_image = app.state.hfer.convert_array_to_base64(image)
        colors = []

    json_str = json.dumps(
        {
            "face_ids": face_ids,
            "colors": colors,
            "image": {"image": annotated_image, "size": image.shape[0:2]},
        }
    )
    return json_str


@app.get("/emotions")
def getEmotionsFromImage(face_id: str):
    face_image = app.state.hfer.get_image_from_id(face_id)
    size = face_image.shape[0:2]
    emotions = app.state.hfer.get_face_emotions_from_image(face_image)
    face_image = app.state.hfer.convert_array_to_base64(face_image)
    json_str = json.dumps(
        {
            "image": {"image": face_image, "size": size},
            "emotions": emotions,
        }
    )
    return json_str
