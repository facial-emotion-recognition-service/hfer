"""Provides an API for extracting faces from an image file."""

import face_recognition


class Extractor:
    def __init__(self):
        pass

    def extract_faces(self, image):
        """Returns the locations of faces from a single image.

        Returns the locations of faces from a single image with the option
        to display the images or save individual faces.

        Args:
            img_path: Path to the image file.

        Returns:
            A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        face_locations = face_recognition.face_locations(image)
        return face_locations
