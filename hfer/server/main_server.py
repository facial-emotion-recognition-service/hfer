import json
from os import makedirs, path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from hfer.server.config_provider import ConfigProvider
from hfer.server.app_logic import AppLogic

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


config = ConfigProvider().config
app.state.hfer = AppLogic(
    config["model_path"],
    config.get("bucket_name"),
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
    """
    API for extracting faces from an image.

    This API processes an input image, detects faces, and returns the results in various formats.
    Faces are identified by UUIDs and can optionally include annotated images and bounding box coordinates.

    Args:
        image (binary file): The image to be processed as binary file.
        include_annotated (bool): Whether to include the annotated image in the response.
        include_bounding_box_coords (bool): Whether to include the bounding box coordinates of the faces in the response.

    Returns:
        dict: A JSON object containing the following keys:
            - face_ids (list of str): A list of UUIDs for the detected faces.
            - If include_annotated is True:
                - colors (list of list of int): RGB color values for the annotations for each face.
                - image (dict): A dictionary containing the annotated image as a base64 string and its size.
            - If include_bounding_box_coords is True:
                - bounding_box_coords (list of list of int): A list of bounding box coordinates for each face, in CSS order.

    Example:

        ``` python
        import requests

        file_path = "path/to/your/image.png"
        url = "http://127.0.0.1:8000/upload_image"

        with open(file_path, "rb") as image_file:
            payload = {"image": image_file}
            # Optional parameters
            params = {
                "include_annotated": True,
                "include_bounding_box_coords": True
            }

            response = requests.post(url, files=payload, params=params, timeout=10)

        print(response.json())

        # To convert the base64 image string from the json response to a PIL Image:
        from PIL import Image

        response_json = json.loads(response.json())
        image_data = response_json["image"]
        image = image_data["image"].encode("latin1")
        size = image_data["size"]
        image = Image.frombytes("RGB", (size[1], size[0]), image)
        ```

        Example Output JSON:
        ```json
        {
            "face_ids": ["75b84592f43a43c8b73f570cf1fe9fb4"],
            "colors": [[153, 51, 51]],
            "image": {
                "image": "BASE64 IMAGE STRING",
                "size": [100, 100]
            },
            "bounding_box_coords": [[31, 69, 74, 26]]
        }
        ```
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
