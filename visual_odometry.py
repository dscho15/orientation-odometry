import cv2
import logging
import time
from pprint import pprint

import numpy as np

from core.camera.camera import PinholeCamera
from core.filters.moving_average import MovingAverageFilter
from core.pose.pose import Pose
from visualizer_2d import visualize_pair
from vo_utils import FeatureTrackingResult
import matplotlib.pyplot as plt

def measure_angle_diff(R1: np.ndarray, R2: np.ndarray) -> float:
    return np.arccos((np.trace(R1.T @ R2) - 1) / 2)

def squared_error(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y, axis=1)

class VisualOdometry(object):
    ransac_tsh_normalized = 2.0
    ransac_confidence = 0.9999
    ransac_method = cv2.RANSAC
    ransac_max_iters = 20000
    keypoint_criteria_min_number_of_frames = 4
    keypoint_criteria_min_movement_tsh = 1
    keypoint_criteria_min_matched_tsh = 20
    essen_mat_eps = 1e-3
    max_rot_eps = np.pi / 4

    def __init__(
        self,
        cam: PinholeCamera,
        fm_extractor: callable,
        matcher: callable,
        do_visualize: bool = True,
    ) -> None:
        self.camera = cam

        self.cur_img = None
        self.prev_img = None

        self.cur_desc = None
        self.prev_desc = None

        self.cur_kps = None
        self.prev_kps = None

        self.fm_extractor = fm_extractor
        self.matcher = matcher

        self.pose_trajectory = [np.eye(4, 4)]

        self.outliers = MovingAverageFilter(10)
        self.time_pose_est = MovingAverageFilter(10)
        self.time_feature_extract = MovingAverageFilter(10)
        self.average_movement = MovingAverageFilter(10)
        self.feature_tracking_results = []

        self.key_results = []
        self.visualize: bool = do_visualize
        self.latest_keyframe_id: int = 0

        logging.basicConfig(level=logging.INFO)

    def check_keypoint_criteria(
        self, prev_kps: np.ndarray, cur_kps: np.ndarray, frame_id: int
    ) -> bool:
        
        avg_movement = np.median(squared_error(prev_kps, cur_kps))
        self.average_movement(avg_movement)
        
        if avg_movement < self.keypoint_criteria_min_movement_tsh:
            return False

        if (frame_id - self.latest_keyframe_id < self.keypoint_criteria_min_number_of_frames):
            return False

        if len(prev_kps) < self.keypoint_criteria_min_matched_tsh:
            return False

        return True

    def drop_outliers(
        self, cur_kps: np.ndarray, prev_kps: np.ndarray, mask: np.ndarray
    ):
        cur_kps = cur_kps[mask.ravel() == 1]
        prev_kps = prev_kps[mask.ravel() == 1]
        return cur_kps, prev_kps

    def find_essential_matrix(
        self, prev_kps: np.ndarray, cur_kps: np.ndarray
    ):
        return cv2.findEssentialMat(
            prev_kps,
            cur_kps,
            cameraMatrix=self.camera.intrinsics,
            maxIters=self.ransac_max_iters,
            method=self.ransac_method,
            prob=self.ransac_confidence,
            threshold=self.ransac_tsh_normalized,
        )

    def decompose_essential_matrix(self, E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R1, R2, t = cv2.decomposeEssentialMat(E)

        if (np.sum(R1.diagonal())) > 3 - self.essen_mat_eps:
            R2 = R1
        elif (np.sum(R2.diagonal())) > 3 - self.essen_mat_eps:
            R1 = R2

        return R1, R2, t

    def estimate_pose(
        self, prev_kps: np.ndarray, cur_kps: np.ndarray
    ):
        pose_timer = time.time()

        E, mask_match = self.find_essential_matrix(prev_kps, cur_kps)
        P = self.recover_pose(E, prev_kps, cur_kps, mask_match)

        self.time_pose_est(time.time() - pose_timer)
        return P, mask_match

    def recover_pose(
        self, E: np.ndarray, prev_kps: np.ndarray, cur_kps: np.ndarray, mask: np.ndarray
    ):
        cur_kps, prev_kps = self.drop_outliers(cur_kps, prev_kps, mask)

        R1, R2, t = self.decompose_essential_matrix(E)

        P1 = np.hstack((R1, t))
        P2 = np.hstack((R1, -t))
        P3 = np.hstack((R2, t))
        P4 = np.hstack((R2, -t))

        inliers, flag = 0, True
        for P in [P1, P2, P3, P4]:
            
            # check if the rotation is not too large
            R = P[:3, :3]
            angle_dif = measure_angle_diff(np.eye(3, 3), R)
            if abs(angle_dif) > self.max_rot_eps:
                continue
            
            prev_proj = self.camera.intrinsics @ np.eye(3, 4)
            
            cur_proj = (
                self.camera.intrinsics
                @ np.eye(3, 4)
                @ np.vstack((P, np.array([0, 0, 0, 1])))
            )

            points_3d = cv2.triangulatePoints(prev_proj, cur_proj, prev_kps.T, cur_kps.T)
            points_3d /= points_3d[3]

            if (points_3d[2] > 0).sum() > inliers:
                inliers = (points_3d[2] > 0).sum()
                valid_pose = np.vstack((P, np.array([0, 0, 0, 1])))
                flag = False
                
        if flag:
            raise Exception("No valid pose found")

        return valid_pose

    def proc_first_frame(self):
        self.keyframe_kps, self.keyframe_desc = self.fm_extractor(self.cur_img)

    def track_features_between_frames(self) -> FeatureTrackingResult:
        feature_timer = time.time()

        self.cur_kps, self.cur_desc = self.fm_extractor(self.cur_img)

        self.matches = self.matcher(self.keyframe_desc, self.cur_desc, self.keyframe_kps, self.cur_kps)

        feature_tracking_result = FeatureTrackingResult(
            self.cur_kps,
            self.cur_desc,
            self.keyframe_kps,
            self.keyframe_desc,
            self.matches,
        )

        self.feature_tracking_results.append(feature_tracking_result)

        self.time_feature_extract(time.time() - feature_timer)
        return feature_tracking_result

    def process_frame(self, i: int, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.cur_img = img

        if i == 0:
            
            self.proc_first_frame()
            self.latest_keyframe_id = i
            self.keyframe_img = img

        else:
            
            self.ft_results = self.track_features_between_frames()

            if not self.check_keypoint_criteria(*(self.ft_results.kps_matched), i):
                
                self.update_history(np.eye(4, 4))

            else:
                
                P, self.mask_match = self.estimate_pose(
                    *(self.ft_results.kps_matched)
                )

                num_matches = self.ft_results.num_matches
                self.num_inliers = self.mask_match.sum()
                self.outliers(self.num_inliers / num_matches)

                self.update_history(P)
                
                if self.visualize:
                    visualize_pair(
                        self.keyframe_img, 
                        self.cur_img, 
                        *(self.ft_results.kps_matched)
                    )

                self.keyframe_img = self.cur_img
                self.keyframe_kps = self.cur_kps
                self.keyframe_desc = self.cur_desc

                self.latest_keyframe_id = i

        return (
            self.outliers.value,
            self.time_pose_est.value,
            self.time_feature_extract.value,
            self.average_movement.value,
        )

    def update_history(self, P: np.ndarray):
        self.pose_trajectory.append(self.pose_trajectory[-1] @ P)