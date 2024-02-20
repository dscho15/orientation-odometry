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

        self.prev_kps = prev_kps
        self.prev_desc = prev_desc
        self.prev_idxs = matches[:, 0]
        
        self.cur_kps = cur_kps
        self.cur_desc = cur_desc
        self.cur_idxs = matches[:, 1]

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

def rotm_2_quat(rotm: np.ndarray):
    qw = np.sqrt(1 + rotm[0, 0] + rotm[1, 1] + rotm[2, 2]) / 2
    if qw < 0:
        qw = -qw
    qx = (rotm[2, 1] - rotm[1, 2]) / (4 * qw)
    qy = (rotm[0, 2] - rotm[2, 0]) / (4 * qw)
    qz = (rotm[1, 0] - rotm[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])

def convert_poses_to_quaternions(
    odom: list[np.ndarray],
):
    rotm_matrices = np.array([pose[:3, :3] for pose in odom])
    quaternions = np.array([rotm_2_quat(rotm) for rotm in rotm_matrices])  
    return quaternions