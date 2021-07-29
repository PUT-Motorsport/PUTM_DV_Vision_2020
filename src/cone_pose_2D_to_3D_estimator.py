#!/usr/bin/env python3

import cv2
import numpy as np
import rospy as rp
import onnxruntime as rt

from typing import List


CONE_MODEL = np.array([
    [0.0,  0.0,     0.325], #K1
    [0.0, -0.038,   0.217], #K2
    [0.0,  0.038,   0.217], #K3
    [0.0, -0.076,   0.108], #K4
    [0.0,  0.076,   0.108], #K5
    [0.0, -0.114,   0.0],   #K6
    [0.0,  0.114,   0.0],   #K7
], dtype = "double")

center = (rp.get_param('/camera_config/image_height') / 2, rp.get_param('/camera_config/image_width') / 2)

focal_length = center[0] / np.tan(60/2 * np.pi / 180)

camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype = "double")

dist_coeffs = np.zeros((4,1))


class ConePoseEstimatorONNX:
    def __init__(self):
        self.model_path = rp.get_param('/models/rektnet/onnx/model_path')

        self.kmodel = rt.InferenceSession(self.model_path)
        self.kinput_name = self.kmodel.get_inputs()[0].name
        self.klabel_name = self.kmodel.get_outputs()[0].name


    def transform(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        cones = np.empty((len(bboxes),3,80,80), np.float32)
        sizes = np.empty((len(bboxes),2))

        for i, (x,y,w,h) in enumerate(bboxes):
            cones[i] = cv2.resize(im[y:y+h, x:x+w], (80, 80)).transpose((2, 0, 1)) * 1./255.0
            sizes[i] = [w,h]

        keypoints = self.kmodel.run([self.klabel_name], {self.kinput_name: cones})[0].astype('double')
        keypoints *= sizes.reshape(-1, 1, 2)
        keypoints += boxes[:, :2].reshape(-1, 1, 2)

        tvecs = []
        for key in keypoints:
            tvec = cv2.solvePnP(
                    objectPoints = CONE_MODEL,
                    imagePoints = key,
                    cameraMatrix = camera_matrix, 
                    distCoeffs = dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )[2]
            tvecs.append(tvec)
            
        tvecs = np.array(tvecs)

        return tvecs


# class ConePoseEstimatorTRT:
#     def __init__(self):
#         self.model_path = rp.get_param('/models/rektnet/trt/model_path')


#     def transform(self, frame: np.ndarray, bboxes: np.ndarray) -> List[Pose]:
#         pass
