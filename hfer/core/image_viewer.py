from PIL import Image, ImageDraw
import face_recognition

class ImageViewer:
    def __init__(self):
        pass

    def display_faces(self, img_path, face_locations):
        img_arr = face_recognition.load_image_file(img_path)
        img = Image.fromarray(img_arr)
        img_draw = ImageDraw.Draw(img)
        for top, right, bottom, left in face_locations:
            rect = [(left, top), (right, bottom)]
            img_draw.rectangle(rect, fill=None, outline=(0, 0, 225))
        # Display the resulting image
        img.show()