#!/usr/bin/env python3

import cv2
import rospy as rp
import numpy as np

from geometry_msgs.msg import Pose

from typing import List


class ConePoseEstimator:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

        self.X_offset = rp.get_param('/sensors/camera_config/X_offset') # mm
        self.Y_offset = rp.get_param('/sensors/camera_config/Y_offset') # mm
        self.FOV = rp.get_param('/sensors/camera_config/FOV') # degrees
        self.base = rp.get_param('/sensors/camera_config/base') # mm
        self.image_width = rp.get_param('/sensors/camera_config/image_width') # px
        
        self.focal_length = (self.image_width / 2) / np.tan(self.FOV/2 * np.pi / 180)


    def calculate_coord_in_right_img(self, cone_roi: np.ndarray, cone_line: np.ndarray, box: List[int], matches_num=5) -> Pose:
        """Calculates coordinates of cone in right image based on coordinates of cone detected in left image.

        Parameters
        ----------
        cone_roi : np.ndarray
            Array with detected cone ROI on left image.
        cone_line : np.ndarray
            Part of right image corresponding to extended ROI coordinates.
        box : List[int]
            Detection coordinates.
        matches_num : int, optional
            Number of matches to estimate cone center, by default 5

        Returns
        -------
        Pose
            Cone pose in space.
        """
        x, _, w, _ = box
        if cone_roi is None or cone_line is None:
            return None

        kp1, des1 = self.sift.detectAndCompute(cv2.cvtColor(cone_roi, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = self.sift.detectAndCompute(cv2.cvtColor(cone_line, cv2.COLOR_BGR2GRAY), None)

        if des1 is None or des2 is None:
            return None

        matches = self.bf.match(des1, des2)
        if len(matches) < 1:
            return None

        matches = sorted(matches, key=lambda x: x.distance)

        right_x_local = [kp2[match.trainIdx].pt[0] for match in matches[:matches_num]]
        right_x_local = np.mean(right_x_local)

        right_x_global = right_x_local + max(0, x - 4*w) - w/2

        disparity = abs(right_x_global - (x+w//2))
        
        return self.calculate_cone_pose(disparity, box)
    

    def calculate_cone_pose(self, disparity: float, box: List[int]) -> Pose:
        """Calculate cone pose in space.

        Parameters
        ----------
        disparity : float
            Distance between cones in left and right images.
        box : List[int]
            Cone coordinates on left image.

        Returns
        -------
        Pose
            Cone pose in space.
        """
        x, y, w, h = box
        pose = Pose()
        
        u = self.image_width/2 - (x + w/2)
        v = self.image_width - (y + h/2)
        
        X = (self.focal_length * self.base / disparity - self.X_offset) / 1000 # m
        Y = X * u / v
        Z = 0.3 # m

        pose.position.x = X
        pose.position.y = Y - self.Y_offset / 1000
        pose.position.z = Z

        return pose
    

    def estimate_cones_poses(self, cone_rois: np.ndarray, cone_lines: np.ndarray, boxes: np.ndarray) -> List[Pose]:
        """Estimates cones position in local racecar space.

        Parameters
        ----------
        cone_rois : np.ndarray
            Array with detected cones ROI on left image.
        cone_lines : np.ndarray
            Parts of right image corresponding to extended ROI coordinates.
        boxes : np.ndarray
            Array with detection coordinates for each detected cone.

        Returns
        -------
        List[Pose]
            List with cones poses.
        """
        cone_poses = [
            self.calculate_coord_in_right_img(cone_roi, cone_line, box, 1) 
            for (cone_roi, cone_line, box) in zip(cone_rois, cone_lines, boxes)]
        
        return cone_poses
