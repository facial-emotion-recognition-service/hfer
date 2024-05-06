import colorsys

import numpy as np
from PIL import Image, ImageDraw


class ImageAnnotator:
    """Class to manage the annotation of the image"""

    def __init__(self):
        pass

    def annotate_faces(self, image, face_coords):
        """Annotate the original image with bounding boxes for each face."""
        image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(image)
        num_faces = len(face_coords)
        colors = self.generate_colors(num_faces)
        min_height = min(bottom - top for top, _, bottom, _ in face_coords)

        for idx in range(num_faces):
            # Draw rectangle
            top, right, bottom, left = face_coords[idx]
            color = colors[idx]
            rect = [(left, top), (right, bottom)]
            img_draw.rectangle(rect, fill=None, outline=color, width=2)

            # Annotate face number
            text = "face" + str(idx + 1)
            font_size = 16 + min(8, (min_height - 75) // 15)
            text_width = img_draw.textlength(text, font_size=font_size)
            rect = [(left, top - font_size), (left + text_width + 2, top)]
            img_draw.rectangle(rect, fill=color, outline=color, width=2)
            text_position = (left + 1, top - font_size - 1)
            img_draw.text(
                text_position, text, fill="black", font_size=font_size
            )

        # Return the array
        image = np.array(image)
        return (image, colors)

    @staticmethod
    def generate_colors(n):
        """Generate n colors with equal perceptual brightness"""
        colors = []
        for i in range(n):
            hue = i * 360.0 / n
            saturation = 0.5
            lightness = 0.4
            rgb_color = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)

            # Convert RGB values to integers in the range 0-255
            rgb_color = tuple(round(c * 255) for c in rgb_color)
            colors.append(rgb_color)
        return colors
