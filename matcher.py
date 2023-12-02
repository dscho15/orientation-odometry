import numpy as np
import cv2

class MatcherBase:
    
    def __init__(self) -> None:
        self.model = None
        
    def __call__(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> np.ndarray:
        NotImplementedError("Implement this method in a child class")
        
class GreedyMatcher(MatcherBase):
    
    def __init__(self, cross_check: bool = True) -> None:
        self.cross_check = cross_check
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, 
                                     crossCheck=self.cross_check)
    
    def __call__(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> np.ndarray:
        matches = self.matcher.match(descriptors1, descriptors2)
        return matches