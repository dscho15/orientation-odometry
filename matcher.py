import numpy as np
import cv2

class FeatureMatcher:

    def __init__(
        self,
        norm_type: int = cv2.NORM_L2,
        ratio_test: float = 0.3,
    ) -> None:
        self.matcher = cv2.BFMatcher(norm_type, False)
        self.matcher_name = "FeatureMatcher"
        self.norm_type = norm_type
        self.ratio_test = ratio_test
        self.fm_ransac_confidence = 0.9999
        self.fm_ransac_reproj_threshold = 1.0
        self.fm_ransac_method = cv2.RANSAC
        self.cross_check = True

    def __call__(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kps1: np.ndarray,
        kps2: np.ndarray,
        mask: np.ndarray = None,
    ):
        indices1, indices2 = [], []

        init_matches1 = self.matcher.knnMatch(desc1, desc2, k=2, mask=mask) # query is desc1, train is desc2
        init_matches2 = self.matcher.knnMatch(desc2, desc1, k=2, mask=mask)

        matches = []
        for i, (m1, n1) in enumerate(init_matches1):
            cond = True
            
            if self.cross_check:
                is_cross_check_valid = (init_matches2[m1.trainIdx][0].trainIdx == i)
                cond *= is_cross_check_valid

            if self.ratio_test is not None:
                is_ratio_test_valid = m1.distance <= self.ratio_test * n1.distance
                cond *= is_ratio_test_valid

            if cond:
                matches.append(m1)
                indices1.append(m1.queryIdx)
                indices2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            points1 = np.array([kps1[m.queryIdx].pt for m in matches])
            points2 = np.array([kps2[m.trainIdx].pt for m in matches])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            points1 = np.array([kps1[m.queryIdx] for m in matches])
            points2 = np.array([kps2[m.trainIdx] for m in matches])
        else:
            raise Exception("kps1 and kps2 must be both list or np.ndarray")
        
        _, mask = cv2.findFundamentalMat(
            points1=points1,
            points2=points2,
            method=self.fm_ransac_method,
            ransacReprojThreshold=self.fm_ransac_reproj_threshold,
            confidence=self.fm_ransac_confidence,
        )

        indices1 = np.array(indices1)[mask.ravel() == 1]
        indices2 = np.array(indices2)[mask.ravel() == 1]

        matches = [(i1, i2) for i1, i2 in zip(indices1, indices2)]
        matches = np.asarray(matches)

        return matches
