import json
from pathlib import Path
from PIL import Image

from hfer.core.predictors import Predictor
from hfer.core.extractor import Extractor
from hfer.core.image_viewer import ImageViewer


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

    def get_face_emotions_from_file(self, face_image_name, top_n, ret):
        img_path = Path(self.image_input_dir, face_image_name)
        result = self.predictor.get_face_image_emotions(img_path, top_n, ret)
        print(result)

        json_str = json.dumps(result, indent=4)
        json_filename = img_path.stem + ".json"
        json_file_path = Path(self.json_output_dir, json_filename)
        with open(json_file_path, "w") as f:
            f.write(json_str)

        return result

    def get_faces_from_file(self, image_file):
        img_path = Path(self.image_input_dir, image_file)
        result = self.extractor.extract_faces(img_path)

        save_dir = Path(self.image_input_dir, "extracted")
        image_stem = Path(image_file).stem

        img = Image.open(img_path)
        face_image_files = []

        for idx, face_coords in enumerate(result):
            top, right, bottom, left = face_coords
            crop_pic = img.crop((left, top, right, bottom))
            image_file = image_stem + "_" + str(idx) + ".jpg"
            save_path = Path(save_dir, image_file)
            crop_pic.save(save_path)

            face_image_files.append(image_file)

        print(face_image_files)

        return face_image_files

    def draw_faces_on_image(self, image_file, face_locations):
        image = self.image_viewer.display_faces(image_file, face_locations)
        return image
