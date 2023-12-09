import numpy as np
from core.pose.pose import Pose
from core.camera.camera import PinholeCamera
import cv2
import logging

from visualizer_2d import visualize_pair


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

        self.cur_kps = cur_kps
        self.cur_desc = cur_desc

        self.prev_idxs = matches[:, 1]
        self.cur_idxs = matches[:, 0]

    @property
    def kps_cur_matched(self):
        return self.cur_kps[self.cur_idxs]

    @property
    def kps_prev_matched(self):
        return self.prev_kps[self.prev_idxs]


class VisualOdometry(object):
    ransac_reproj_tsh = 0.1
    confidence = 0.999

    def __init__(self, cam, fm_extractor, matcher, gt=None) -> None:
        self.cam = cam

        self.cur_img = None
        self.prev_img = None

        self.cur_desc = None
        self.prev_desc = None

        self.cur_kp = None
        self.prev_kp = None

        self.fm_extractor = fm_extractor
        self.matcher = matcher

        logging.basicConfig(level=logging.INFO)

        self.cur_pose = Pose()

    def absolute_scale(self, frame_id: int) -> np.ndarray:
        return np.r_[0, 0, 1]

    def find_relative_scale(self, frame_id):
        pass

    def extract_features(self, img: np.ndarray) -> (np.ndarray, np.ndarray):
        kp1, desc1 = self.fm_extractor(img)
        return kp1, desc1

    def remove_outliers(
        self, cur_kp: np.ndarray, prev_kp: np.ndarray, mask: np.ndarray
    ):
        if mask is None:
            return cur_kp, prev_kp

        cur_kp = cur_kp[mask.ravel() == 1]
        prev_kp = prev_kp[mask.ravel() == 1]

        return cur_kp, prev_kp

    def filter_matches(
        self, matches: np.ndarray, cur_kp: np.ndarray, prev_kp: np.ndarray
    ):
        cur_kp = cur_kp[matches[:, 0]]
        prev_kp = prev_kp[matches[:, 1]]

        return cur_kp, prev_kp

    def estimate_pose(
        self, cur_kp: np.ndarray, prev_kp: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        E, self.mask_match = cv2.findEssentialMat(
            cur_kp,
            prev_kp,
            cameraMatrix=self.cam.intrinsics,
            method=cv2.USAC_MAGSAC,
            prob=self.confidence,
            threshold=self.ransac_reproj_tsh,
        )

        _, R, t, self.mask_match = cv2.recoverPose(
            E, cur_kp, prev_kp, focal=1.0, pp=(0.0, 0.0), mask=self.mask_match
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
            self.ft_results = self.track_features_between_frames()

            # estimate pose
            R, t = self.estimate_pose(
                self.ft_results.kps_cur_matched, self.ft_results.kps_cur_matched, False
            )

            # update variables
            self.cur_kps = self.ft_results.cur_kps
            self.cur_desc = self.ft_results.cur_desc
            
            self.prev_kps = self.ft_results.prev_kps
            self.prev_desc = self.ft_results.prev_desc
            
            self.num_matches = self.ft_results.matches.shape[0]
            self.num_inliers = self.sum(self.mask_match)
            
            
            

            # update history with the new pose estimation
            self.update_history(R, t)

            # find relative scale
            self.find_relative_scale(i)

        if i > 10:
            exit()

    def update_history(self):
        pass


if __name__ == "__main__":
    VisualOdometry("")
