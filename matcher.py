import numpy as np
import cv2


class MatcherBase:
    def __init__(self) -> None:
        self.model = None

    def __call__(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray
    ) -> np.ndarray:
        NotImplementedError("Implement this method in a child class")


class GreedyMatcher(MatcherBase):
    def __init__(self, cross_check: bool = True) -> None:
        self.cross_check = cross_check
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=self.cross_check)

    def __call__(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray
    ) -> np.ndarray:
        matches = self.matcher.match(descriptors1, descriptors2)
        matches = [np.array([m.queryIdx, m.trainIdx]) for m in matches]
        matches = np.array(matches)
        return matches


class RatioMatcher(MatcherBase):
    def __init__(self, ratio: float = 0.7) -> None:
        self.ratio = ratio
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def __call__(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray
    ) -> np.ndarray:
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        matches = [
            np.array([m[0].queryIdx, m[0].trainIdx])
            for m in matches
            if m[0].distance < self.ratio * m[1].distance
        ]
        matches = np.array(matches)
        return matches
