from typing import Any
import cv2
import numpy as np


class SiftExtractor:
    def __init__(self) -> None:
        # create sift featuer extractor
        self.sift = cv2.SIFT_create()

    def __call__(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract sift features from image
        """
        # compute sift features
        keypoints, descriptors = self.sift.detectAndCompute(image, mask)

        # convert keypoints to numpy array
        keypoints = np.array([kp.pt for kp in keypoints])
        descriptors = np.array(descriptors)

        return (keypoints, descriptors)


class OrbExtractor:
    def __init__(self) -> None:
        # create sift featuer extractor
        self.orb = cv2.ORB.create(2000)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract sift features from image
        """
        # compute sift features
        keypoints, descriptors = self.orb.detectAndCompute(image, mask)

        # convert keypoints to numpy array
        keypoints = np.array([kp.pt for kp in keypoints])
        descriptors = np.array(descriptors)

        return (keypoints, descriptors)


class RootSiftExtractor:
    def __init__(self) -> None:
        # create sift featuer extractor
        self.sift = cv2.SIFT_create()

    def __call__(
        self, image: np.ndarray, mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract sift features from image
        """
        # compute sift features
        keypoints, descriptors = self.sift.detectAndCompute(image, mask)

        # convert keypoints to numpy array
        keypoints = np.array([kp.pt for kp in keypoints])
        descriptors = np.array(descriptors)

        # normalize descriptors
        if descriptors is not None:
            descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
            descriptors = np.sqrt(descriptors)

        return (keypoints, descriptors)


if __name__ == "__main__":
    image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
    extractor = SiftExtractor()
    kp, desc = extractor(image)
    print(kp, desc)
