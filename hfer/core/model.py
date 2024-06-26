"""Provides an abstraction of the model for the rest of the application.

Provides API for interacting with the trained model, including loading and
preprocessing images, and making predictions.
"""

import numpy as np
import tensorflow as tf
from google.cloud import storage

LABELS_TEXT2NUM = {
    "surprise": 0,
    "fear": 1,
    "disgust": 2,
    "happiness": 3,
    "anger": 4,
    "sadness": 5,
    "neutral": 6,
}


def preprocess_image(image: np.array) -> np.array:
    """Preprocesses an image.

    Given an np.array, resizes it to the
    expected input size of the pretrained model, and preprocesses it to the
    format expected by the model (scaling, mean subtraction, RGB to BGR,
    etc. as applicable).

    Args:
        img_path: Full path to the image file.

    Returns:
        A numpy array containing the preprocessed image.
    """

    image = tf.keras.preprocessing.image.array_to_img(image).resize((224, 224))
    image_preprocessed = preprocess(tf.keras.preprocessing.image.img_to_array(image))

    return image_preprocessed


def preprocess(face_image):
    """Preprocesses a numpy array containing a right-sized image of a face.

    Given a numpy array containing an image of a face, preprocesses it to
    the format expected by the model (scaling, mean subtraction, RGB to BGR,
    etc. as applicable).

    Args:
        face_image: A numpy array containing an image of a face, resized to
            the expected input size of the pretrained model.

    Returns:
        A numpy array containing the preprocessed image.
    """
    img_array = np.expand_dims(face_image, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)

    return img_array


class Model:
    def __init__(self, model_path, bucket_name):
        self._model = self.load_model(model_path, bucket_name)
        self.labels_text2num = LABELS_TEXT2NUM
        self.labels_num2text = {v: k for k, v in self.labels_text2num.items()}

    def predict(self, img_array):
        """Gets predictions from the model for an already-preprocessed image.

        Args:
            img_array: Image as a numpy array, preprocessed to suit the model.

        Returns:
            The model's predictions for the image as a numpy array containing an
            array of the probabilities for each emotion label.
        """
        return self._model.predict(img_array)[0]

    def load_model(self, model_path, bucket_name):
        """Loads the model from the model path. If a GCS bucket name is provided,
        will first download the most recent model from the bucket and save it
        locally at model path"""
        if bucket_name:
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blobs = bucket.list_blobs(prefix="models/")

            try:
                latest_blob = max(blobs, key=lambda x: x.updated)
                latest_blob.download_to_filename(model_path)
            except:
                print(f"\nNo model found in GCS bucket {bucket_name}")
                return None

        return tf.keras.models.load_model(model_path, compile=False)
