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
        self.image_height = rp.get_param('/sensors/camera_config/image_height') # px
        self.center = (self.image_width/2, self.image_width/2)
        
        self.focal_length = (self.image_width / 2) / np.tan(self.FOV/2 * np.pi / 180)

        self.box_model = np.array([
            [0.0,  0.114, 0.325],
            [0.0, -0.114, 0.325],
            [0.0,  0.114, 0.0],
            [0.0, -0.114, 0.0]
        ], dtype = "double")

        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype = "double")

        self.dist_coeffs = np.zeros((4,1))


    def calculate_coord_in_right_img(self, cone_roi: np.ndarray, cone_line: np.ndarray, box: List[int], matches_num=5) -> Pose:
        """Calculates coordinates of conein right image based on coordinates of cone detected in left image.

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
        kp1, des1 = self.sift.detectAndCompute(cv2.cvtColor(cone_roi, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = self.sift.detectAndCompute(cv2.cvtColor(cone_line, cv2.COLOR_BGR2GRAY), None)

        if des1 is None or des2 is None:
            return None

        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        matches = []
        for m in knn_matches:
            if len(m) > 1:
                if m[0].distance < m[1].distance * 0.75:
                    matches.append(m[0])
            else:
                matches.append(m[0])

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
        v = self.image_height - (y + h/2)
        
        X = (self.focal_length * self.base / disparity - self.X_offset) / 1000 # m
        Y = X * u / v
        Z = 0.3 # m

        pose.position.x = X
        pose.position.y = Y - self.Y_offset / 1000
        pose.position.z = Z

        return pose
    

    def estimate_cones_poses(self, cone_rois: np.ndarray, right_img: np.ndarray, boxes: np.ndarray) -> List[Pose]:
        """Estimates cones position in local racecar space.

        Parameters
        ----------
        cone_rois : np.ndarray
            Array with detected cones ROI on left image.
        right_img : np.ndarray
            Image from right camera.
        boxes : np.ndarray
            Array with detection coordinates for each detected cone.

        Returns
        -------
        List[Pose]
            List with cones poses.
        """
        cone_lines = self.propagate_bounding_boxes(boxes, right_img)
        cone_poses = [
            self.calculate_coord_in_right_img(cone_roi, cone_line, box) 
            for (cone_roi, cone_line, box) in zip(cone_rois, cone_lines, boxes)]
        
        return cone_poses

    def propagate_bounding_boxes(self, boxes: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Propagates bounding boxes to right image.

        Parameters
        ----------
        boxes : np.ndarray
            Array with detection coordinates for each detected cone.
        right_img : np.ndarray
            Image from right camera.

        Returns
        -------
        List[Pose]
            List with cones poses.
        """

        cone_lines = []
        for box in boxes:
            box_points = np.array([
                    (box[0], box[1]),
                    (box[0] + box[2], box[1]),
                    (box[0], box[1] + box[3]),
                    (box[0] + box[2], box[1] + box[3])
                ], dtype="double")

            tvec = cv2.solvePnP(
                objectPoints = self.box_model,
                imagePoints = box_points,
                cameraMatrix = self.camera_matrix, 
                distCoeffs = self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )[2]
            d = (0.12 * self.focal_length) / tvec[2]

            right_box = [
                (box_points[0][0] - d[0], box_points[0][1]),
                (box_points[1][0] - d[0], box_points[1][1]),
                (box_points[2][0] - d[0], box_points[2][1]),
                (box_points[3][0] - d[0], box_points[3][1])
            ]

            cone_lines.append(
                right_img[
                    max(0, int(right_box[0][1])):min(self.image_height, int(right_box[2][1])),
                    max(0, int(right_box[0][0])):min(self.image_width, int(right_box[1][0])),
                    ::-1
                ])

        return cone_lines
