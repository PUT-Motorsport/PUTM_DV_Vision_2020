#!/usr/bin/env python3

import cv2
import yaml
import rospy as rp
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray

from cone_detector import ConeDetector
from cone_classifier import ConeClassifier
from cone_pose_estimator import ConePoseEstimator


class ImageInference:
    def __init__(self):
        self.detector = ConeDetector()
        self.classifier = ConeClassifier()
        self.pose_estimator = ConePoseEstimator()
        

    def inference(self, left_img: np.ndarray, right_img: np.ndarray):
        pass


    def publish_cones_position(self, yellow_cones, blue_cones):
        yellow_pose_array = PoseArray()
        blue_pose_array = PoseArray()

        yellow_pose_array.header.stamp = rp.Time.now()
        blue_pose_array.header.stamp = rp.Time.now()
        yellow_pose_array.header.frame_id = 'putm/cones_poses'
        blue_pose_array.header.frame_id = 'putm/cones_poses'

        yellow_pose_array.poses = yellow_cones
        blue_pose_array.poses = blue_cones

        self.yellow_cones_position_publisher.publish(yellow_pose_array)
        self.blue_cones_position_publisher.publish(blue_pose_array)


    def publish_inferenced_img(self, img: np.ndarray, boxes: np.ndarray, cone_colors: np.ndarray):
        for box, cone_color in zip(boxes, cone_colors):
            x, y, w, h = box.astype(int)
            cv2.rectangle(img, (x,y), (x+w, y+h), self.colors[cone_color], 2)

        img_msg = Image()
        img_msg.header.stamp = rp.Time.now()
        img_msg.header.frame_id = "/putm/vision/inferenced_image"
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = img.flatten().tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        self.inferenced_img_pub.publish(img_msg)


if __name__ == '__main__':
    rp.init_node('vision', log_level=rp.DEBUG)

    img_infer = ImageInference()

    while not rp.is_shutdown():
        rp.spin()