#!/usr/bin/env 

import os 
import time
import numpy as np
import rospy as rp
import argparse

import cv2 
import pycuda.autoinit

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import get_input_shape, TrtYOLO

WINDOW_NAME = "Test"

class ConeDetectorTRT:
    def __init__(self):
        self.conf_th = 0.3

        self.model = "yolov4-tiny-416"
        if not os.path.isfile('yolo/%s.trt' % self.model):
            raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % self.model)

        h, w = get_input_shape(self.model)
        self.trt_yolo = TrtYOLO(self.model, (h, w), 1, False)

        # self.cam = add_camera_args(/dev/video1)
        # Below 
        parser = argparse.ArgumentParser()
        parser = add_camera_args(parser)
        args = parser.parse_args()
        cam = Camera(args)
        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        self.predict_debug(cam)
        cam.release()
        cv2.destroyAllWindows()

    def predict_debug(self, cam):
        """ This needs to be changed to something that is returning
            the boxes array so it can be used in the pipeline
        """
        fps = 0.0
        tic = time.time()
        while True:
            # cam.read() gets frame from camera, basically a cv2.VideoCapture with
            # abstraction
            img = cam.read()
            if img is None:
                break
            #  boxes, confs, clss = self.trt_yolo.detect(img, conf_th)
            boxes = self.predict(img)
            print(boxes)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            print(fps)

    def predict(self, img):
        boxes, _, _ = self.trt_yolo.detect(img, self.conf_th)
        return np.array(boxes)

            
if __name__ == '__main__':
    detect = ConeDetectorTRT()
