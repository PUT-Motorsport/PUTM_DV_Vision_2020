#!/usr/bin/env 

import os 
import time
import yaml
import numpy as np
import rospy as rp

import cv2 
import pycuda.autoinit

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import get_input_shape, TrtYOLO

class ConeDetectorTRT:
    def __init__(self):
        self.model = "yolov4-tiny-416"
        self.h, self.w = get_input_shape(self.model)
        self.trt_yolo = (self.model, (self.h, self.w), 1, False)

        # self.cam = add_camera_args(/dev/video1)
        self.cam = add_camera_args

        
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

if __name__ = '__main__':
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    h, w = get_input_shape(args.model)
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()
    detect = ConeDetectorTRT()

