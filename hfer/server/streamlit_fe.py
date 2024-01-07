import streamlit as st
import json
import requests
from hfer.server.app_config_provider import AppConfigProvider

st.title("Human Facial Emotion Recognizer")

st.write("")
st.write(
    "Upload an image of an isolated face \
    This will be updated to accept any \
    image."
)

st.header("Try it out!")
image_file = st.file_uploader("Upload an image of a face", type=["png"])
## I don't actually know if our model accepts more types?

if image_file is not None:
    st.image(image_file.read(), caption="Uploaded image")
    response = requests.get(url="http://127.0.0.1:8000/")
    st.write(response)
    payload = {"image_file": (image_file.name, image_file.read())}
    response = requests.post(url="http://127.0.0.1:8000/upload_image", files=payload)
    st.write(response)
    # f.write(image_file.getbuffer())
    # st.success("Saved File")
    # top_three = dict(sorted(predictions.items(), key=lambda x: -x[1])[:3])
    # st.header("Your Results")
    # for l, p in top_three.items():
    #    st.subheader(l)
    #    st.write("Probability: " + str(round(p*100, 1)) + "%")
