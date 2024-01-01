import numpy as np


class FeatureTrackingResult(object):
    def __init__(
        self,
        cur_kps: np.ndarray,
        cur_desc: np.ndarray,
        prev_kps: np.ndarray,
        prev_desc: np.ndarray,
        matches: np.ndarray,
    ):
        assert (
            prev_kps.shape[0] == prev_desc.shape[0]
        ), "Number of keypoints and descriptors for previous frame do not match"

        assert (
            cur_kps.shape[0] == cur_desc.shape[0]
        ), "Number of keypoints and descriptors for current frame do not match"

        assert matches.shape[1] == 2, "Matches array should have shape (n, 2)"

        self.cur_kps = cur_kps
        self.cur_desc = cur_desc
        self.cur_idxs = matches[:, 1]

        self.prev_kps = prev_kps
        self.prev_desc = prev_desc
        self.prev_idxs = matches[:, 0]

    @property
    def kps_cur_matched(self):
        return self.cur_kps[self.cur_idxs]

    @property
    def kps_prev_matched(self):
        return self.prev_kps[self.prev_idxs]

    @property
    def kps_matched(self):
        return self.kps_prev_matched, self.kps_cur_matched

    @property
    def num_matches(self):
        return self.cur_idxs.shape[0]

    @property
    def results(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.cur_kps,
            self.cur_desc,
            self.cur_idxs,
            self.prev_kps,
            self.prev_desc,
            self.prev_idxs,
        )
