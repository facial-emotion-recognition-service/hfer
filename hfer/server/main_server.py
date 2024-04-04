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
def uploadImage(image_file: UploadFile, sub_folder: str = "raw"):
    save_dir = path.join(app.state.hfer.image_input_dir, sub_folder)
    if not path.exists(save_dir):
        makedirs(save_dir)

    file_location = path.join(save_dir, image_file.filename)

    with open(file_location, "wb") as f:
        f.write(image_file.file.read())
    return {
        "INFO": f"File '{image_file.filename}' saved to your {file_location}."
    }


@app.get("/image")
def getImage(image_path: str):
    print("get image")
    json_str = app.state.hfer.get_image(image_path, "json")
    return json_str


@app.get("/faces_from_image")
def getFaceImages(image_path: str):
    json_str = app.state.hfer.get_faces_from_file(image_path)

    return json_str


@app.get("/emotions_from_image")
def getEmotionsFromImage(image_path: str):
    # print("Server.getEmotionsFromImage.name = " + face_image_file)
    json_str = app.state.hfer.get_face_emotions_from_file(image_path, 8, "text")
    # return HttpResponse("getEmotionsFromImage " + face_image_file)
    return json_str
