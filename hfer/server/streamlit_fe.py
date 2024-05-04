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
    payload = {"image": file_content}

    ## Upload the file and save it to the back-end specified location
    response = requests.post(
        url="http://127.0.0.1:8000/upload_image", files=payload, timeout=10
    )

    ## I think this is acually just returning a dict, not json
    response_json = response.json()
    img_data = response_json["annotated_image"].encode("latin1")
    ## this was hard coded for testing, should actually pass this back
    img = Image.frombytes("RGB", (1200, 1355), img_data)
    st.image(img, caption="Uploaded image")

    ## Currently returns each face as a binary object
    ## If we refactor to the api should look like:
    ## (image, n_prediction) -> ((tuple of coords), (dict of predicts))

    face_ids = response_json["face_ids"]
    st.header(f'{len(face_ids)} face{"" if len(face_ids) == 1 else "s"} detected.')

    table = "| Face | Emotion Predictions (Probability) |"
    table += (
        "\n| --- | --- |\n"
        if len(response_json) == 1
        else " Face | Emotion Predictions (Probability) |\n| --- | --- | --- | --- |\n"
    )
    for i, face_id in enumerate(face_ids):
        response = requests.get(
            url="http://127.0.0.1:8000/emotions",
            params={"face_id": face_id},
            timeout=10,
        )
        response_json = response.json()
        predictions = response_json["emotions"]

        ## put in nice table

        top_three = dict(
            sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        )

        ## this is currently incorrect and should be fixed, need to store
        ## image size somehwere and pass that along with the image
        img_data = response_json["image"].encode("latin1")
        img = Image.frombytes("RGB", (100, 100), img_data)

        # Add image to the table
        img_data_uri = get_image_data_uri(img)
        table += f"<img src='{img_data_uri}' width='50'> | "

        # Add predictions to the table
        for l, p in top_three.items():
            table += f"{l.title()} (" + str(round(p * 100, 1)) + "%)<br>"

        table += " |\n" if i % 2 == 1 else " | "

    # Display the table
    st.markdown(table, unsafe_allow_html=True)
