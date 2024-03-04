import numpy as np
import cv2


class FeatureMatcher:

    def __init__(
        self,
        norm_type: int = cv2.NORM_HAMMING,
        ratio_test: float = 0.25,
    ) -> None:
        
        self.matcher = cv2.BFMatcher(norm_type, False)
        self.matcher_name = "FeatureMatcher"
        self.norm_type = norm_type
        self.ratio_test = ratio_test
        self.distance = 20

    def __call__(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kps1: np.ndarray,
        kps2: np.ndarray,
        mask: np.ndarray = None,
    ):
        init_matches1 = self.matcher.knnMatch(desc1, desc2, k=2)
        init_matches2 = self.matcher.knnMatch(desc2, desc1, k=2)

        indices1, indices2 = [], []
        for (m1, n1) in init_matches1:
            
            m2 = init_matches2[m1.trainIdx][0]
            is_cross_check_valid = (m2.trainIdx == m1.queryIdx)
            
            if not is_cross_check_valid:
                continue
            
            dist = kps1[m1.queryIdx] - kps2[m1.trainIdx]
            dist = np.sqrt(np.linalg.norm(dist))
            if dist > self.distance:
                continue

            is_ratio_test_valid = m1.distance <= self.ratio_test * n1.distance
            if not is_ratio_test_valid:
                continue

            indices1.append(m1.queryIdx)
            indices2.append(m1.trainIdx)
        
        matches = []
        for i1, i2 in zip(indices1, indices2):
            matches.append((i1, i2))
            
        matches = np.array(matches)

        return matches
