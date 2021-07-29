#!/usr/bin/env python3

import cv2
import yaml
import rospy as rp
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray

from cone_detector import ConeDetectorOpenCV as ConeDetector
from cone_pose_2D_to_3D_estimator import ConePoseEstimatorONNX as ConePoseEstimator

from typing import List


class MonoCamera:
    def __init__(self):
        self.detector = ConeDetector()
        self.pose_2D_to_3D_estimator = ConePoseEstimator()
        
        self.colors = [(0,0,255), (0,255,255), (255,0,0)]


    def main(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1200)
        cap.set(5, 60)

        while not rp.is_shutdown():
            ret, frame = cap.read()

            # detect
            bboxes, scores, classes = self.detector.detect(frame)
            bboxes = np.array(bboxes)

            # rois = np.array([frame[y:y+h, x:x+w] for (x,y,w,h) in bboxes])

            # transform 2D to 3D
            cones_poses_3D = self.pose_2D_to_3D_estimator.transform(frame, bboxes)

            # publish detections
            self.publish_cones_poses(cones_poses_3D, classes)

            # publish inferenced image # only for testing
            # self.publish_inferenced_img(frame, bboxes, classes)


    def publish_cones_poses(self, cones_poses: List[Pose], classes: np.ndarray):
        red_pose_array = PoseArray()
        yellow_pose_array = PoseArray()
        blue_pose_array = PoseArray()

        for color_id, pose_array in enumerate([red_pose_array, yellow_pose_array, blue_pose_array]):
            pose_array.header.stamp = rp.Time.now()
            pose_array.poses = cones_poses[classes == color_id]

        self.red_cones_position_publisher.publish(red_pose_array)
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
    rp.init_node('vision_mono', log_level=rp.DEBUG)

    mono = MonoCamera()

    mono.main()

    rp.spin()