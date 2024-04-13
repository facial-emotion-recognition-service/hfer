import base64
import json
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


def get_image_data_uri(image: Image.Image):
    """
    Converts an image to a data URI in JPEG format.

    Parameters:
    - img (PIL.Image.Image): The image to be converted.

    Returns:
    - str: The image data URI in JPEG format.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return "data:image/jpeg;base64," + img_str


st.title("Human Facial Emotion Recognizer")

st.write("")
st.write("Upload an image. This app will find the faces and identify the emotions.")

st.header("Try it out!")
image_file = st.file_uploader("Upload an image of a face", type=["png", "jpg"])
## I don't actually know if our model accepts more types?

if image_file is not None:
    file_content = image_file.read()
    st.image(file_content, caption="Uploaded image")
    payload = {"image": file_content, "top_n": 3, 'include_coordinates_in_results' = False}

    ## Upload the file and save it to the back-end specified location
    response = requests.post(
        url="http://127.0.0.1:8000/upload_image", files=payload, timeout=10
    )

    ## Currently returns each face as a binary object
    ## If we refactor to the api should look like:
    ## (image, n_prediction) -> ((tuple of coords), (dict of predicts))
    response = requests.get(
        url="http://127.0.0.1:8000/faces_from_image",
        params={"image_path": os.path.join("raw", image_file.name)},
        timeout=10,
    )
    response_json = response.json()
    st.header(
        f'{len(response_json)} face{"" if len(response_json) == 1 else "s"} detected.'
    )

    table = "| Face | Emotion Predictions (Probability) |"
    table += (
        "\n| --- | --- |\n"
        if len(response_json) == 1
        else " Face | Emotion Predictions (Probability) |\n| --- | --- | --- | --- |\n"
    )
    for i, face_image_file in enumerate(response_json):
        response = requests.get(
            url="http://127.0.0.1:8000/emotions_from_image",
            params={"image_path": os.path.join("extracted", face_image_file)},
            timeout=10,
        )
        predictions = response.json()

        ## put in nice table

        top_three = dict(
            sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        )

        response = requests.get(
            url="http://127.0.0.1:8000/image",
            params={"image_path": os.path.join("extracted", face_image_file)},
            timeout=10,
        )
        json_str = response.json()
        image_info = json.loads(json_str)
        img_data = image_info["data"].encode("latin1")
        img = Image.frombytes(image_info["mode"], image_info["size"], img_data)

        # Add image to the table
        img_data_uri = get_image_data_uri(img)
        table += f"<img src='{img_data_uri}' width='50'> | "

        # Add predictions to the table
        for l, p in top_three.items():
            table += f"{l.title()} (" + str(round(p * 100, 1)) + "%)<br>"

        table += " |\n" if i % 2 == 1 else " | "

    # Display the table
    st.markdown(table, unsafe_allow_html=True)
