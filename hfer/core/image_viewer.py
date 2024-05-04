from PIL import Image, ImageDraw
import numpy as np


class ImageViewer:
    def __init__(self):
        pass

    def display_faces(self, image, face_coords):
        image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(image)
        for top, right, bottom, left in face_coords:
            rect = [(left, top), (right, bottom)]
            img_draw.rectangle(rect, fill=None, outline=(0, 0, 225))
        # return the resulting image
        image = np.array(image)
        return image
