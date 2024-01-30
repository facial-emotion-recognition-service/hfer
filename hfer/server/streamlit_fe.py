import streamlit as st
import json
import requests
from hfer.server.app_config_provider import AppConfigProvider
import os

st.title("Human Facial Emotion Recognizer")

st.write("")
st.write(
    "Upload an image of an isolated face \
    This will be updated to accept any \
    image."
)

st.header("Try it out!")
image_file = st.file_uploader("Upload an image of a face", type=["png", "jpg"])
## I don't actually know if our model accepts more types?

if image_file is not None:
    file_content = image_file.read()
    st.image(file_content, caption="Uploaded image")
    payload = {"image_file": (image_file.name, file_content)}

    ## Upload the file and save it to the back-end specified location
    response = requests.post(url="http://127.0.0.1:8000/upload_image", files=payload)

    response = requests.get(
        url="http://127.0.0.1:8000/emotions_from_image",
        params={"image_path": os.path.join("raw", image_file.name)},
    )
    predictions = response.json()

    ## Display predictions
    top_three = dict(sorted(predictions.items(), key=lambda x: -x[1])[:3])
    st.header("Your Results")
    for l, p in top_three.items():
        st.subheader(l)
        st.write("Probability: " + str(round(p * 100, 1)) + "%")
