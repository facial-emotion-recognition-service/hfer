import json
from pathlib import Path
from io import BytesIO
import uuid

from PIL import Image
import numpy as np

from hfer.core.extractor import Extractor
from hfer.core.image_viewer import ImageViewer
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
        self.image_viewer = ImageViewer()
        self.image_input_dir = Path(image_input_dir)
        self.json_output_dir = Path(json_output_dir)
        self.faces_dict = {}

    def get_face_emotions_from_image(self, image: np.array, top_n=3, ret="text"):
        result = self.predictor.get_face_image_emotions(image, top_n, ret)
        return result

    def get_faces_from_image(self, image: np.array):

        result = self.extractor.extract_faces(image)
        face_ids = []

        for face_coords in result:
            top, right, bottom, left = face_coords
            crop_pic = image[top:bottom, left:right]

            face_id = uuid.uuid4().hex
            face_ids.append(face_id)

            self.faces_dict[face_id] = (crop_pic, face_coords)

        return face_ids

    def get_image(self, face_image_name, _type=None):
        ## Consider using this and passing this around instead of the image path
        img_path = Path(self.image_input_dir, face_image_name)
        img = Image.open(img_path)
        if _type == "json":
            # Create a dictionary to store image information
            image_info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "data": img.tobytes().decode("latin1"),  # Convert bytes to string
            }

            # Convert dictionary to JSON
            json_str = json.dumps(image_info)
            return json_str

        return img

    def get_annotated_image(self, image, face_ids):
        face_coords = [self.faces_dict[face_id][1] for face_id in face_ids]
        annotated_image = self.image_viewer.display_faces(image, face_coords)
        return annotated_image

    def convert_upload_to_array(self, image) -> np.array:
        image = BytesIO(image.file.read())
        image = Image.open(image)
        image = np.array(image)
        return image

    def convert_array_to_base64(self, image: np.array) -> str:
        image = (
            Image.fromarray(np.uint8(image)).convert("RGB").tobytes().decode("latin1")
        )
        return image

    def get_image_from_id(self, face_id) -> np.array:
        image = self.faces_dict.get(face_id)[0]
        return image
