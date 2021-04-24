#!/usr/bin/env 

import os 
import time
import yaml
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
        parser = argparse.ArgumentParser()
        self.model = "yolov4-tiny-416"
        if not os.path.isfile('yolo/%s.trt' % self.model):
            raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % self.model)

        h, w = get_input_shape(self.model)
        cls_dict = ["Cone"]
        vis = BBoxVisualization(cls_dict)
        trt_yolo = TrtYOLO(self.model, (h, w), 1, False)

        # self.cam = add_camera_args(/dev/video1)
        parser = add_camera_args(parser)
        args = parser.parse_args()
        self.cam = Camera(args)
        if not self.cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        open_window(
            WINDOW_NAME, 'Camera TensorRT YOLO Demo',
            self.cam.img_width, self.cam.img_height)
        self.loop_and_detect(self.cam, trt_yolo, conf_th=0.3, vis=vis)
        self.cam.release()
        cv2.destroyAllWindows()
        
    def loop_and_detect(self, cam, trt_yolo, conf_th, vis):
        """Continuously capture images from camera and do object detection.

        # Arguments
          cam: the camera instance (video source).
          trt_yolo: the TRT YOLO object detector instance.
          conf_th: confidence/score threshold for object detection.
          vis: for visualization.
        """
        full_scrn = False
        fps = 0.0
        tic = time.time()
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break
            img = cam.read()
            if img is None:
                break
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
            elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
                full_scrn = not full_scrn
                set_display(WINDOW_NAME, full_scrn)

if __name__ == '__main__':
    detect = ConeDetectorTRT()
