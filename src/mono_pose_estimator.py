#!/usr/bin/env python3

import cv2
import rospy as rp
import numpy as np
import onnxruntime as rt

class ConeMonoEstimator:
    def __init__(self):
        self.kmodel = rt.InferenceSession(rp.get_param('/sensors/vision/rektnet/model'))
        self.kinput_name = self.kmodel.get_inputs()[0].name
        self.klabel_name = self.kmodel.get_outputs()[0].name
        self.cone_size = int(rp.get_param('/sensors/vision/rektnet/size'))

        self.FOV = rp.get_param('/sensors/mono_config/FOV')
        self.image_width = rp.get_param('/sensors/mono_config/image_width')

        self.focal_length = (self.image_width / 2) / np.tan(self.FOV/2 * np.pi / 180)
        self.center = (self.image_width/2, self.image_width/2)

        self.cone_model = np.array([
                            [0.0,  0.0,     0.325], #K1
                            [0.0, -0.038,   0.217], #K2
                            [0.0,  0.038,   0.217], #K3
                            [0.0, -0.076,   0.108], #K4
                            [0.0,  0.076,   0.108], #K5
                            [0.0, -0.114,   0.0],   #K6
                            [0.0,  0.114,   0.0],   #K7
                        ], dtype = "double")

        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]], dtype = "double"
            )

        self.dist_coeffs = np.zeros((4,1))

    def predict(self, boxes: np.ndarray, img: np.ndarray):
        boxes = np.array(boxes)

        boxes = boxes[boxes[:, 2]*boxes[:, 3]>100]

        cones = np.empty((boxes.shape[0],3,self.cone_size,self.cone_size), np.float32)
        sizes = np.empty((boxes.shape[0],2))

        for i, (x,y,w,h) in enumerate(boxes):
            cones[i] = cv2.resize(img[y:y+h, x:x+w], (self.cone_size, self.cone_size)).transpose((2, 0, 1)) * 1./255.0
            sizes[i] = [w,h]

        keypoints = self.kmodel.run([self.klabel_name], {self.kinput_name: cones})[0].astype('double')
        keypoints *= sizes.reshape(-1, 1, 2)
        keypoints += boxes[:, :2].reshape(-1, 1, 2)

        tvecs = []
        for key in keypoints:
            tvec = cv2.solvePnP(
                    objectPoints = self.cone_model,
                    imagePoints = key,
                    cameraMatrix = self.camera_matrix, 
                    distCoeffs = self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )[2]

            tvecs.append(tvec)
            
        return np.array(tvecs)