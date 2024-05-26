import json
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from hfer.core.extractor import Extractor
from hfer.core.image_annotator import ImageAnnotator
from hfer.core.predictors import Predictor

## TO DO Roll app config provider into app_logic
## TO DO roll model config provider into core.model??
## TO DO fix core.image_viewer so it returns an image
## Rename image_viwere


class AppLogic:
    def __init__(
        self,
        model_path,
        image_input_dir,
        json_output_dir,
        config_data,
        bucket_name,
    ):
        self.predictor = Predictor(model_path, config_data, bucket_name)
        self.extractor = Extractor()
        self.image_annotator = ImageAnnotator()
        self.image_input_dir = Path(image_input_dir)
        self.json_output_dir = Path(json_output_dir)
        self.faces_dict = {}

    def get_face_emotions_from_image(self, image: np.array, top_n=3, ret="text"):
        result = self.predictor.get_face_image_emotions(image, top_n, ret)
        return result

    def get_faces_from_image(self, image: np.array):

        face_coords = self.extractor.extract_faces(image)
        # Sort the faces as a human would sort them. (top to bottom, left to right)
        # The boxes are put in 10 horizontal bands and sorted from left to right.
        # x[0] and x[3] are the top and left values of the bounding box, respectively.
        height = image.shape[0]
        face_coords = sorted(face_coords, key=lambda x: (x[0] // (height / 10), x[3]))
        face_ids = []

        for face_coord in face_coords:
            top, right, bottom, left = face_coord
            crop_pic = image[top:bottom, left:right]

            face_id = uuid.uuid4().hex
            face_ids.append(face_id)

            self.faces_dict[face_id] = (crop_pic, face_coord)

        return face_ids, face_coords

    def get_annotated_image(self, image, face_ids):
        face_coords = [self.faces_dict[face_id][1] for face_id in face_ids]
        annotated_image, colors = self.image_annotator.annotate_faces(
            image, face_coords
        )
        return annotated_image, colors

    def convert_upload_to_array(self, image) -> np.array:
        image = BytesIO(image.file.read())
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        return image

    def convert_array_to_base64(self, image: np.array) -> str:
        image = (
            Image.fromarray(np.uint8(image)).convert("RGB").tobytes().decode("latin1")
        )
        return image

    def get_image_from_id(self, face_id) -> np.array:
        image = self.faces_dict.get(face_id)[0]
        if image.any():
            self.faces_dict.pop(face_id)
        return image
