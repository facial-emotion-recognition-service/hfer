# Human Facial Emotion Recognition (HFER)

## Introduction
HFER is a Python project that leverages convolutional neural networks to detect emotions from human faces which are extracted from images uploaded by a user. It can recognize the following seven emotions: surprise, fear, disgust, happiness, anger, sadness, and neutral. We used multiclass image classification, a type of supervised deep learning algorithm. This is the code repo for the back-end implementation. For further details on the model, see the project's main [README](https://github.com/facial-emotion-recognition-service).

## Usage
### REST API
HFER's REST API was written using FastAPI and the container has been deployed on Google Cloud Run which can be accessed [here](https://hfer-api-3s6mpd7w3q-uw.a.run.app/).

**Note:** Cold starts are enabled by default on GCR, so the container may take up to a few minutes to start the first time it is accessed.

#### How to use the API
<details>
<summary>❗️❗️❗️ TO DO ❗️❗️❗️</summary>
Step-by-step guide to using the API and a link to the API documentation once generated
</details>

### Web UI
HFER has a web interface written using Streamlit. The code resides in its own separate repo [here](https://github.com/facial-emotion-recognition-service/hfer_front). It is currently deployed [here](https://hfer-farid-nathan.streamlit.app/).

## Installation (For Development):
### Install Locally
1. Clone the repo.
2. Download the trained model file [here](https://drive.google.com/file/d/1EXQdc-XM1vzkO4KLeSbUMfJk9w-rvehG/view?usp=drive_link) and copy it into the `models` directory.
3. Install the `hfer` package in editable mode with:
   ``` bash
   pip install -e .
   ```
4. From the `hfer/hfer` directory run:
   ``` bash
   uvicorn hfer.server.main_server:app --reload
   ```
### Install Using Docker
1. Follow steps 1-2 above.
2. Build the docker image with:
   ``` bash
   docker build -t hfer-api .
   ```
3. Run the container with:
   ``` bash
   docker run -p 8000:8000 hfer-api
   ```

> **Note:** If you would like to deploy the front-end locally as well, follow the instructions in the [`hfer_front` repo](https://github.com/facial-emotion-recognition-service/hfer_front)'s README.

## Contributors
HFER was developed by two friends, [Farid](https://github.com/artificialfintelligence) and [Nathan](https://github.com/nihonlanguageprocessing), with significant input and guidance from a third friend: [Or](https://github.com/orbartal).
