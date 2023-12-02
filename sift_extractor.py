from typing import Any
import cv2
import numpy as np

class SiftExtractor:
    
    def __init__(self) -> None:
        
        # create sift featuer extractor
        self.sift = cv2.SIFT_create()
    
    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Any:
        """
        Extract sift features from image
        """
        # compute sift features
        keypoints, descriptors = self.sift.detectAndCompute(image, mask)
        return keypoints, descriptors
    
if __name__ == "__main__":
    
    image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
    extractor = SiftExtractor()
    kp, desc = extractor(image)
    print(kp, desc)