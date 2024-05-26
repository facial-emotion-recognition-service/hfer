import json
from os import makedirs, path

from fastapi import FastAPI, UploadFile, HTTPException
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
    app_config.get("bucket_name"),
)


# Define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


@app.post("/upload_image")
def uploadImage(
    image: UploadFile,
    include_annotation: bool = True,
    include_bounding_box_coords: bool = True,
):
    """API that takes an image and extracts the faces from it.

    Retrieves the faces as UUIDs in temporary storage.

    Args:
        image: Image to be processed.
        include_annotated: Include the annotated image in the response.
        include_bounding_box_coords: Includes the bounding box of the faces in the response.

    Returns:
        A json with the face_ids as UUIDs by default.
        The include annotated and include bounding box coords give options
        to include more data in the return.
    """

    # Extract faces from image
    image = app.state.hfer.convert_upload_to_array(image)
    face_ids, bounding_boxes = app.state.hfer.get_faces_from_image(image)

    results = {"face_ids": face_ids}

    # Get annotated image
    if include_annotation:
        # Case where faces were found in uploaded image
        if face_ids:
            annotated_image, colors = app.state.hfer.get_annotated_image(
                image, face_ids
            )
            annotated_image = app.state.hfer.convert_array_to_base64(annotated_image)
        # Case where faces weren't found
        else:
            annotated_image = app.state.hfer.convert_array_to_base64(image)
            colors = []
        results["colors"] = colors
        results["image"] = {"image": annotated_image, "size": image.shape[0:2]}

    if include_bounding_box_coords:
        results["bounding_box_coords"] = bounding_boxes

    json_str = json.dumps(results)

    return json_str


@app.get("/emotions")
def getEmotions(
    face_id: str = None,
    face_image: UploadFile = None,
    include_image: bool = False,
    top_n=3,
    ret="text",
):
    """API that returns the top-n emotions from a single face.

    Retrieves the top n emotions from an image of a single, isolated face,
    along with their probabilities. The face can be retrieved from temporary
    storage by UUID or passed directly to the API.

    Args:
        face_id: UUID for the face image.
        face_image: Face image
        include_image: Include the face image in the response.
        top_n: Number of top emotions to return.
        ret: Label type for the returned dict. One of "text" or "num".

    Returns:
        A json with the top-n emotions for the face.
    """

    # Get the face image in the correct format.
    # Either retrieve from the temporary storage or process as np_array.
    if face_id:
        face_image = app.state.hfer.get_image_from_id(face_id)
    elif face_image:
        face_image = app.state.hfer.convert_upload_to_array(face_image)
    else:
        raise HTTPException(
            status_code=404,
            detail="Please provide one of either a face_id as a UUID or an image of an isolated face.",
        )

    emotions = app.state.hfer.get_face_emotions_from_image(face_image, top_n, ret)

    # Prepare the results.
    results = {"emotions": emotions}
    if include_image:
        size = face_image.shape[0:2]
        face_image = app.state.hfer.convert_array_to_base64(face_image)
        results["image"] = {"image": face_image, "size": size}

    json_str = json.dumps(results)
    return json_str
