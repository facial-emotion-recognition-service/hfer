from os import path

from hfer.server.app_config_provider import AppConfigProvider
from hfer.server.app_logic import AppLogic
from hfer.server.model_config_provider import ModelConfigProvider

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

## streamlit run hfer/server/streamlit_fe.py
## run in hfer
## uvicorn hfer.server.main_server:app --reload
## run in hfer/hfer

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
    file_location = path.join(
        app.state.hfer.image_input_dir, sub_folder, image_file.filename
    )
    print(file_location)
    with open(file_location, "wb") as f:
        f.write(image_file.file.read())
    return {"INFO": f"File '{image_file.filename}' saved to your {file_location}."}


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
