import numpy as np
import cv2


class FeatureMatcher:

    def __init__(
        self,
        norm_type: int = cv2.NORM_L2,
        ratio_test: float = 0.50,
        fm_ransac_max_iters: int = 1000000,
        fm_ransac_confidence: float = 0.999,
        fm_ransac_reproj_threshold: float = 3.0,
        fm_ransac_method: int = cv2.FM_RANSAC
    ) -> None:
        
        self.matcher = cv2.BFMatcher(norm_type, False)
        self.matcher_name = "FeatureMatcher"
        self.norm_type = norm_type
        self.ratio_test = ratio_test
        self.fm_ransac_confidence = fm_ransac_confidence
        self.fm_ransac_reproj_threshold = fm_ransac_reproj_threshold
        self.fm_ransac_method = fm_ransac_method
        self.fm_ransac_max_iters = fm_ransac_max_iters

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

            if self.ratio_test:
                is_ratio_test_valid = m1.distance <= self.ratio_test * n1.distance
                if not is_ratio_test_valid:
                    continue

            indices1.append(m1.queryIdx)
            indices2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            points1 = np.array([kps1[m] for m in indices1])
            points2 = np.array([kps2[m] for m in indices2])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            points1 = np.array([kps1[m] for m in indices1])
            points2 = np.array([kps2[m] for m in indices2])
        else:
            raise Exception("kps1 and kps2 must either be lists or np.ndarrays")
        
        try:
            F, mask = cv2.findFundamentalMat(
                points1=points1,
                points2=points2,
                method=self.fm_ransac_method,
                ransacReprojThreshold=self.fm_ransac_reproj_threshold,
                confidence=self.fm_ransac_confidence
            )
            if mask is None:
                raise Exception("Mask is None")
        except:
            raise Exception("Failed to estimate fundamental matrix")
        
        matches = []
        for m, i1, i2 in zip(mask, indices1, indices2):
            if m:
                matches.append((i1, i2))
        matches = np.array(matches)

        return matches
