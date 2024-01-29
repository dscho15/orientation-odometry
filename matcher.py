import numpy as np
import cv2
import logging
from enum import Enum

class FeatureMatcher:
    ratio_test = 0.5
    fm_ransac_confidence = 0.999
    fm_ransac_reproj_threshold = 1.0
    fm_ransac_method = cv2.RANSAC

    def __init__(
        self,
        norm_type: int = cv2.NORM_HAMMING,
        cross_check: bool = False,
        ratio_test: float = 0.5,
    ) -> None:
        self.cross_check = cross_check
        self.matcher = cv2.BFMatcher(norm_type, cross_check)
        self.matcher_name = "FeatureMatcher"
        self.norm_type = norm_type
        self.ratio_test = ratio_test

    def __call__(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kps1: np.ndarray,
        kps2: np.ndarray,
        mask: np.ndarray = None,
    ):
        indices1, indices2 = [], []

        init_matches1 = self.matcher.knnMatch(desc1, desc2, k=2, mask=mask)
        init_matches2 = self.matcher.knnMatch(desc2, desc1, k=2, mask=mask)

        logging.info(
            f"{self.matcher_name}: {len(init_matches1)} initial matches \t ⊂(◉‿◉)つ"
        )

        matches = []
        for i, (m1, n1) in enumerate(init_matches1):
            cond = True
            
            if self.cross_check:
                is_cross_check_valid = init_matches2[m1.trainIdx][0].trainIdx == i
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
        n_inlier = np.count_nonzero(mask)

        logging.info(
            f"{self.matcher_name}: {n_inlier}/{len(matches)} ({n_inlier/len(matches)*100:.2f}%) inliers \t (ㆆ _ ㆆ)"
        )

        indices1 = np.array(indices1)[mask.ravel() == 1]
        indices2 = np.array(indices2)[mask.ravel() == 1]

        matches = [(i1, i2) for i1, i2 in zip(indices1, indices2)]
        matches = np.asarray(matches)

        return matches
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    fm = FeatureMatcher()

    # load random image online
    img1 = cv2.imread('datasets/2011_09_26/2011_09_26_drive_0018_extract/image_00/data/0000000000.png') # queryImage
    img2 = cv2.imread('datasets/2011_09_26/2011_09_26_drive_0018_extract/image_00/data/0000000192.png') # trainImage

    # extract orb features
    orb = cv2.ORB_create(8000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # convert des1, des2 to ndarray
    des1 = np.asarray(des1)
    des2 = np.asarray(des2)

    kp1 = [kp for kp in np.asarray(kp1)]
    kp2 = [kp for kp in np.asarray(kp2)]

    # match features
    matches = fm(des1, des2, kp1, kp2)
