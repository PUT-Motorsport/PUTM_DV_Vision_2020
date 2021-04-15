#!/usr/bin/env python3

import os
import cv2
import yaml 
import numpy as np
import rospy as rp


class ConeDetector:
    def __init__(self):
        self.yolo_weights_file = rp.get_param('/sensors/vision/yolo/yolo_weights_file')
        self.yolo_config_file = rp.get_param('/sensors/vision/yolo/yolo_config_file')
        self.confidence_threshold = rp.get_param('/sensors/vision/yolo/confidence_threshold')
        self.nms_threshold = rp.get_param('/sensors/vision/yolo/nms_threshold')
        self.model_size = rp.get_param('/sensors/vision/yolo/model_size')

        self.net = cv2.dnn.readNetFromDarknet(self.yolo_config_file, self.yolo_weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.model_size, self.model_size), scale=1/255)

        if self.net is None: 
            print("Error loading cone detector model")


    def predict(self, img: np.ndarray) -> np.ndarray:
        """Detects cones in image.

        Parameters
        ----------
        img : np.ndarray
            Image data array.

        Returns
        -------
        np.ndarray
            Array which contain detection boxes in form [x,y,w,h].
        """
        _, _, boxes = self.model.detect(img, confThreshold=self.confidence_threshold, nmsThreshold=self.nms_threshold)

        return np.array(boxes)
