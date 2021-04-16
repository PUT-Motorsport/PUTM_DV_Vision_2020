#!/usr/bin/env python3

import cv2
import numpy as np
import rospy as rp
import tensorflow as tf


class ConeClassifier:
    def __init__(self):
        self.model_path = rp.get_param('/sensors/vision/squeezenet/squeezenet_model_path')
        self.model_input_shape = rp.get_param('/sensors/vision/squeezenet/model_input_shape')

        self.model = tf.keras.models.load_model(self.model_path)

        if self.model is None: 
            print("Error loading cone classifier model")


    def predict(self, rois: np.ndarray) -> np.ndarray:
        """Predict cone class.

        Parameters
        ----------
        rois : np.ndarray
            Array with regions of interest detected by object detector.

        Returns
        -------
        np.ndarray
            Classification labels for each ROI in rois array.
        """
        rois = [cv2.resize(roi, (self.model_input_shape[1], self.model_input_shape[0])) for roi in rois]
        rois = np.array(rois).astype(np.float32) / 255

        preds = self.model.predict(rois)
        labels = np.argmax(preds, axis=1)

        return labels
