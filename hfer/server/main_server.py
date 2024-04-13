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
def uploadImage(image: UploadFile, top_n: int = 3, sub_folder: str = "raw"):
    ## get the image
    img = image.read()
    ## conver this to np array
    ## image = numpy.array(Image.open(io.BytesIO(image_bytes)))

    ##
    faces = app.state.hfer.get_faces_from_image(image)
    for face in faces:
    app.state.hfer.get_emotions_form_image(image, top_n)



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
