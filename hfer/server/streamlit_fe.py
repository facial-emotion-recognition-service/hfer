import base64
import json
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


def convert_base64_to_pil(image_data) -> Image.Image:
    """
    Converts a string represented as a base64 string to a PIL image.

    Parameters:
    - image_data (dict{'image', 'size'}
    - 'image' (string): The image to be converted.
    - 'size' (tuple(int, int)): The size of the image.

    Returns:
    - img (PIL.Image.Image): The image.
    """

    image = image_data["image"].encode("latin1")
    size = image_data["size"]
    image = Image.frombytes("RGB", (size[1], size[0]), image)
    return image


st.title("Human Facial Emotion Recognizer")

st.write("")
st.write("Upload an image. This app will find the faces and identify the emotions.")

st.header("Try it out!")
image_file = st.file_uploader("Upload an image of a face", type=["png", "jpg"])
## I don't actually know if our model accepts more types?

if image_file is not None:
    file_content = image_file.read()
    payload = {"image": file_content}

    response = requests.post(
        url="http://127.0.0.1:8000/upload_image", files=payload, timeout=10
    )

    response_json = json.loads(response.json())
    img = convert_base64_to_pil(response_json["image"])

    st.image(img, caption="Uploaded image")

    face_ids = response_json["face_ids"]
    colors = response_json["colors"]
    st.header(f'{len(face_ids)} face{"" if len(face_ids) == 1 else "s"} detected.')

    # Everything below this line just renders the table of extracted faces and
    # emotions as a markdown table. Images are inserted in HTML tags.
    table = ""
    if face_ids:
        table += "| Face | Emotion Predictions (Probability) |"
        table += (
            "\n| --- | --- |\n"
            if len(face_ids) == 1
            else " Face | Emotion Predictions (Probability) |\n| --- | --- | --- | --- |\n"
        )
    for i, face_id in enumerate(face_ids):
        response = requests.get(
            url="http://127.0.0.1:8000/emotions",
            params={"face_id": face_id, "include_image": True},
            timeout=10,
        )
        response_json = json.loads(response.json())
        predictions = response_json["emotions"]

        ## put in nice table

        top_three = dict(
            sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        )

        img = convert_base64_to_pil(response_json["image"])

        # Add image to the table
        img_data_uri = get_image_data_uri(img)
        table += f"<span alignment='center' style='color:rgb{tuple(colors[i])};'>face{i+1}</span><br>"
        table += f"<img src='{img_data_uri}' width='50'>"
        table += " | "

        # Add predictions to the table
        for l, p in top_three.items():
            table += f"{l.title()} (" + str(round(p * 100, 1)) + "%)<br>"

        table += " |\n" if i % 2 == 1 else " | "

    # Display the table
    st.markdown(table, unsafe_allow_html=True)
