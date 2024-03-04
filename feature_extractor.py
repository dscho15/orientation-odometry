from typing import Any
import cv2
import numpy as np

from functools import partial


class CV2Extractor:
    def __init__(self) -> None:
        self.model = None

    def __call__(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints, descriptors = self.model.detectAndCompute(image, mask)

        keypoints = np.array([kp.pt for kp in keypoints])
        descriptors = np.array(descriptors)

        return (keypoints, descriptors)


class SiftExtractor(CV2Extractor):

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.model = cv2.SIFT_create(nfeatures=n_features,
                                     contrastThreshold=0.01)


class RootSiftExtractor(CV2Extractor):

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.model = cv2.SIFT_create(nfeatures=n_features,
                                     contrastThreshold=0.01)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints, descriptors = self.model.detectAndCompute(image, mask)

        keypoints = np.array([kp.pt for kp in keypoints])
        descriptors = np.array(descriptors)

        if descriptors is not None:
            descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
            descriptors = np.sqrt(descriptors)

        return (keypoints, descriptors)


class OrbExtractor(CV2Extractor):
    def __init__(self, n_features: int) -> None:
        super().__init__
        self.model = cv2.ORB.create(n_features)


class GoodFeaturesToTrackExtractor:

    def __init__(self, n_features: int) -> None:
        self.good_features_to_track = partial(
            cv2.goodFeaturesToTrack,
            maxCorners=n_features,
            qualityLevel=0.01,
            minDistance=20,
        )

    def __call__(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints = self.good_features_to_track(image).reshape(-1, 2)
        return (keypoints, None)
