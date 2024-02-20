from typing import Any
import cv2
import numpy as np

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
    def __init__(self) -> None:
        super().__init__()
        self.model = cv2.SIFT_create(nfeatures=8000)

class RootSiftExtractor(CV2Extractor):
    def __init__(self) -> None:
        super().__init__()
        self.model = cv2.SIFT_create(nfeatures=8000)

    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints, descriptors = self.model.detectAndCompute(image, mask)

        keypoints = np.array([kp.pt for kp in keypoints])
        descriptors = np.array(descriptors)

        if descriptors is not None:
            descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
            descriptors = np.sqrt(descriptors)

        return (keypoints, descriptors)

class OrbExtractor(CV2Extractor):
    def __init__(self, n_features: int=3000) -> None:
        super().__init__
        self.model = cv2.ORB.create(n_features)
