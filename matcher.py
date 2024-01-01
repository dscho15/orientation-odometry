import numpy as np
import cv2
import logging
from enum import Enum

class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1     

ratio_test = 0.5

def feature_matcher_factory(norm_type: int=cv2.NORM_HAMMING, cross_check: bool=False, ratio_test: float=ratio_test, type=FeatureMatcherTypes.BF):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    return None 

class FeatureMatcher:
    def __init__(self, 
                 norm_type = cv2.NORM_HAMMING, 
                 cross_check: bool = False,
                 ratio_test: float = 0.5,
                 type: FeatureMatcherTypes = FeatureMatcherTypes.BF) -> None:
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.matches = []
        self.ratio_test = ratio_test 
        self.matcher = None 
        self.matcher_name = ''

    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> np.ndarray:
        matches = self.matcher.match(descriptors1, descriptors2)
        self.matches = matches
        return self.good_matches(descriptors1, descriptors2)

    def good_matches(self, desc1: np.ndarray, desc2: np.ndarray, kps1: np.ndarray, kps2: np.ndarray, ratio_test: float=None, cross_check: bool=True, err_thld: int=1):
        
        idx1, idx2 = [], []      
        if ratio_test is None: 
            ratio_test = self.ratio_test
            
        init_matches1 = self.matcher.knnMatch(desc1, desc2, k=2)
        init_matches2 = self.matcher.knnMatch(desc2, desc1, k=2)

        good_matches = []

        for i,(m1,n1) in enumerate(init_matches1):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[m1.trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio_test is not None:
                cond2 = m1.distance <= ratio_test * n1.distance
                cond *= cond2
            if cond:
                good_matches.append(m1)
                idx1.append(m1.queryIdx)
                idx2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            good_kps1 = np.array([kps1[m.queryIdx].pt for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx].pt for m in good_matches])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            good_kps1 = np.array([kps1[m.queryIdx] for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx] for m in good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        ransac_method = None 
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC
        _, mask = cv2.findFundamentalMat(good_kps1, good_kps2, ransac_method, err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        return idx1, idx2, good_matches, mask
        
class BfFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=ratio_test, type = FeatureMatcherTypes.BF):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher = cv2.BFMatcher(norm_type, cross_check)     
        self.matcher_name = 'BfFeatureMatcher'
