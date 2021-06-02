#!/usr/bin/env python3

import cv2
import yaml
import rospy as rp
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
import message_filters

from cone_detector import ConeDetector
from cone_classifier import ConeClassifier
from cone_pose_estimator import ConePoseEstimator
from mono_pose_estimator import ConeMonoEstimator


class ImageInference:
    def __init__(self):
        self.detector = ConeDetector()
        self.classifier = ConeClassifier()
        self.pose_estimator = ConePoseEstimator()
        self.mono_estimator = ConeMonoEstimator()

        right_sub = message_filters.Subscriber("/fsds/camera/cam1", Image) # right camera
        left_sub = message_filters.Subscriber("/fsds/camera/cam2", Image) # left camera
        mono_sub = message_filters.Subscriber("/fsds/camera/cam_mono", Image) # mono camera

        ts = message_filters.ApproximateTimeSynchronizer([right_sub, left_sub, mono_sub], queue_size=5, slop=0.1)
        ts.registerCallback(self.camera_image_callback)

        self.inferenced_img_pub = rp.Publisher("/putm/vision/inferenced_image", Image, queue_size=100)
        self.yellow_cones_position_publisher = rp.Publisher('/putm/vision/yellow_cones_position', PoseArray, queue_size=10)
        self.blue_cones_position_publisher = rp.Publisher('/putm/vision/blue_cones_position', PoseArray, queue_size=10)
        self.mono_cones_position_publisher = rp.Publisher('/putm/vision/mono_cones_position', PoseArray, queue_size=10)

        self.image_width = rp.get_param('/sensors/camera_config/image_width') # px

        self.colors = [(0,0,255), (0,255,255), (255,0,0)] # [red, yellow, blue] in BGR format

        self.inference_step = 2
        self.frame = 0


    def camera_image_callback(self, right_img_data: Image, left_img_data: Image, mono_img_data):
        right_img = np.frombuffer(right_img_data.data, dtype=np.uint8).reshape(right_img_data.height, right_img_data.width, -1)
        left_img = np.frombuffer(left_img_data.data, dtype=np.uint8).reshape(left_img_data.height, left_img_data.width, -1)
        mono_img = np.frombuffer(mono_img_data.data, dtype=np.uint8).reshape(mono_img_data.height, mono_img_data.width, -1)
        self.frame += 1

        if self.frame % self.inference_step == 0:
            self.inference(left_img, right_img)
            self.mono_camera_inference(mono_img)

    def mono_camera_inference(self, img: np.ndarray): 
        
        boxes = np.array(self.detector.predict(img))

        tvecs = self.mono_estimator.predict(boxes, img)

        cones = []
        for cone in tvecs:
            pose = Pose()
            pose.position.x = np.negative(cone[1])
            pose.position.y = np.negative(cone[0])
            pose.position.z = 0.0

            cones.append(pose)

        mono_pose_array = PoseArray()
        mono_pose_array.header.stamp = rp.Time.now()
        mono_pose_array.header.frame_id = 'putm/cones_poses'
        mono_pose_array.poses = cones
        self.mono_cones_position_publisher.publish(mono_pose_array)


    def inference(self, left_img: np.ndarray, right_img: np.ndarray):
        boxes = self.detector.predict(left_img)

        if len(boxes) >= 1:
            cone_rois = [left_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], ::-1] for box in boxes]
            cone_lines = [
                right_img[box[1]-int(0.25*box[3]):box[1]+int(1.25*box[3]), 
                        max(0, box[0]-4*box[2]):min(self.image_width, box[0]+box[2]+4*box[2]), ::-1] for box in boxes]

            cone_colors = self.classifier.predict(cone_rois)
            cone_poses = self.pose_estimator.estimate_cones_poses(cone_rois, cone_lines, boxes)

            cones = list(filter(lambda x: x[1] is not None, zip(cone_colors, cone_poses)))
            yellow_cones = [cone[1] for cone in cones if cone[0] == 1]
            blue_cones = [cone[1] for cone in cones if cone[0] == 2]

            self.publish_cones_position(yellow_cones, blue_cones)
            self.publish_inferenced_img(left_img, boxes, cone_colors)


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