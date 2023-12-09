import numpy as np
from core.pose.pose import Pose
from core.camera.camera import PinholeCamera
import cv2
import logging
import time

from visualizer_2d import visualize_pair
from core.filters.moving_average import MovingAverageFilter


class FeatureTrackingResult(object):
    def __init__(
        self,
        cur_kps: np.ndarray,
        cur_desc: np.ndarray,
        prev_kps: np.ndarray,
        prev_desc: np.ndarray,
        matches: np.ndarray,
    ) -> None:
        assert (
            prev_kps.shape[0] == prev_desc.shape[0]
        ), "Number of keypoints and descriptors for previous frame do not match"
        assert (
            cur_kps.shape[0] == cur_desc.shape[0]
        ), "Number of keypoints and descriptors for current frame do not match"
        assert matches.shape[1] == 2, "Matches array should have shape (n, 2)"

        self.prev_kps = prev_kps
        self.prev_desc = prev_desc
        self.prev_idxs = matches[:, 1]

        self.cur_kps = cur_kps
        self.cur_desc = cur_desc
        self.cur_idxs = matches[:, 0]

    @property
    def kps_cur_matched(self):
        return self.cur_kps[self.cur_idxs]

    @property
    def kps_prev_matched(self):
        return self.prev_kps[self.prev_idxs]

    @property
    def num_matches(self):
        return self.cur_idxs.shape[0]


class VisualOdometry(object):
    ransac_reproj_tsh = 0.1
    confidence = 0.999

    def __init__(self, cam, fm_extractor, matcher) -> None:
        self.cam = cam

        self.cur_img = None
        self.prev_img = None

        self.cur_desc = None
        self.prev_desc = None

        self.cur_kps = None
        self.prev_kps = None

        self.fm_extractor = fm_extractor
        self.matcher = matcher

        self.cur_pose = Pose()
        self.pose_history = []

        self.outliers = MovingAverageFilter(10)
        self.time_pose_est = MovingAverageFilter(10)
        self.time_feature_extract = MovingAverageFilter(10)

        logging.basicConfig(level=logging.INFO)

    def absolute_scale(self, frame_id: int) -> np.ndarray:
        return np.r_[0, 0, 1]

    def find_relative_scale(self, frame_id):
        pass

    def extract_features(self, img: np.ndarray) -> (np.ndarray, np.ndarray):
        kp1, desc1 = self.fm_extractor(img)
        return kp1, desc1

    def remove_outliers(
        self, cur_kps: np.ndarray, prev_kps: np.ndarray, mask: np.ndarray = None
    ):
        if mask is not None:
            cur_kps = cur_kps[mask.ravel() == 1]
            prev_kps = prev_kps[mask.ravel() == 1]
        return cur_kps, prev_kps

    def estimate_pose(
        self, cur_kps: np.ndarray, prev_kps: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        E, self.mask_match = cv2.findEssentialMat(
            cur_kps,
            prev_kps,
            cameraMatrix=self.cam.intrinsics,
            method=cv2.USAC_MAGSAC,
            prob=self.confidence,
            threshold=self.ransac_reproj_tsh,
        )

        _, R, t, self.mask_match = cv2.recoverPose(
            E, cur_kps, prev_kps, focal=1.0, pp=(0.0, 0.0), mask=self.mask_match
        )

        return (R, t)

    def proc_first_frame(self):
        self.cur_kps, self.cur_desc = self.extract_features(self.cur_img)

    def track_features_between_frames(self) -> FeatureTrackingResult:
        self.cur_kps, self.cur_desc = self.extract_features(self.cur_img)
        self.matches = self.matcher(self.cur_desc, self.prev_desc)
        feature_tracking_result = FeatureTrackingResult(
            self.cur_kps, self.cur_desc, self.prev_kps, self.prev_desc, self.matches
        )
        return feature_tracking_result

    def process_frame(self, i: int, img: np.ndarray):
        self.prev_img = self.cur_img
        self.cur_img = img
        self.prev_kps, self.prev_desc = self.cur_kps, self.cur_desc

        if i == 0:
            self.proc_first_frame()
        else:
            # match features
            feature_timer = time.time()
            self.ft_results = self.track_features_between_frames()
            self.time_feature_extract(time.time() - feature_timer)

            # estimate pose
            pose_timer = time.time()
            R, t = self.estimate_pose(
                self.ft_results.kps_cur_matched, self.ft_results.kps_prev_matched
            )
            self.time_pose_est(time.time() - pose_timer)

            # update variables
            self.cur_kps = self.ft_results.cur_kps
            self.cur_desc = self.ft_results.cur_desc

            self.prev_kps = self.ft_results.prev_kps
            self.prev_desc = self.ft_results.prev_desc

            self.num_matches = self.ft_results.num_matches
            self.num_inliers = self.mask_match.sum()
            self.outliers(self.num_inliers / self.num_matches)

            # update history with the new pose estimation
            # self.update_history(R, t)

            # find relative scale
            # self.find_relative_scale(i)

        return (
            self.outliers.value,
            self.time_pose_est.value,
            self.time_feature_extract.value,
        )

    def update_history(self):
        pass


if __name__ == "__main__":
    VisualOdometry("")
