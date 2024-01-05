"""Provides an API for getting predictions from the model in various formats.

Provides an API for getting various types of pre-formatted predictions from the
underlying model for various types of input data.
"""
import hfer.core.model as model


class Predictor:
    def __init__(self, model_path, config_data, bucket_name):
        self.model = model.Model(model_path, config_data, bucket_name)

    def get_face_image_emotions(self, face_image_file, top_n=1, ret="text"):
        """Returns the top n emotions for an image of a single, isolated face.

        Retrieves the top n emotions from an image of a single, isolated face,
        along with their probabilities.

        Args:
            face_image_file: Path to the face image file.
            top_n: Number of top emotions to return.
            ret: Label type for the returned dict. One of "text" or "num".

        Returns:
            A dict mapping the top n emotions to their probabilities.
        """
        img_array = model.preprocess_file(face_image_file)
        result = self._get_face_emotions(img_array, top_n, ret)

        return result

    # TODO(https://trello.com/c/p9RyBsxE): Refactor once extraction is done.
    # This is an "internal" (private, or rather: protected) method put in place
    # only for compatibility with Nathan's WiP on extraction. To be refactored
    # and merged with get_face_image_emotions().

    def _get_face_emotions(self, face, top_n=1, ret="text"):
        predictions = self.model.predict(face)
        preds_sorted = sorted(predictions, reverse=True)
        preds_sorted_indices = [
            i
            for i, _ in sorted(
                enumerate(predictions), key=lambda x: x[1], reverse=True
            )
        ]
        top_n_preds_num = preds_sorted_indices[:top_n]
        top_n_preds_text = list(
            map(lambda x: self.model.labels_num2text[x], top_n_preds_num)
        )
        dict_labels = top_n_preds_text if ret == "text" else top_n_preds_num
        result = {
            label: float(preds_sorted[i]) for i, label in enumerate(dict_labels)
        }

        return result
