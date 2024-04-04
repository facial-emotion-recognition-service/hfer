import json
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.title("Human Facial Emotion Recognizer")

st.write("")
st.write(
    "Upload an image. This app will find the faces and identify the emotions."
)

st.header("Try it out!")
image_file = st.file_uploader("Upload an image of a face", type=["png", "jpg"])
## I don't actually know if our model accepts more types?

if image_file is not None:
    file_content = image_file.read()
    st.image(file_content, caption="Uploaded image")
    payload = {"image_file": (image_file.name, file_content)}

    ## Upload the file and save it to the back-end specified location
    response = requests.post(
        url="http://127.0.0.1:8000/upload_image", files=payload, timeout=10
    )

    response = requests.get(
        url="http://127.0.0.1:8000/faces_from_image",
        params={"image_path": os.path.join("raw", image_file.name)},
        timeout=10,
    )
    response_json = response.json()
    st.header(
        f'{len(response_json)} face{"" if len(response_json) == 1 else "s"} detected.'
    )

    faces_dict = {"faces": []}

    for face_image_file in response_json:
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
        faces_dict["faces"].append(img)
        st.image(img)
        for l, p in top_three.items():
            st.write(
                f"{l.title()}: Probability: " + str(round(p * 100, 1)) + "%"
            )

    print(faces_dict)
    faces_df = pd.DataFrame(faces_dict)
    st.data_editor(
        faces_dict,
        column_config={
            "faces": st.column_config.ImageColumn(
                "Preview Image", help="Streamlit app preview screenshots"
            )
        },
        hide_index=True,
    )
