import numpy as np
import cv2

class PinholeCamera:
    
    def __init__(self, 
                 intrinsics: np.array = None,
                 dist_params: np.array = None) -> None:
        self.intrinsics = intrinsics
        self.dist_params = dist_params
    
    def project(self, 
                points_3d: np.array) -> np.array:
        """
        Project 3D points to 2D image plane
        """
        pass
    
    def undistort(self, 
                  img: np.array) -> np.array:
        """
        Undistort image
        """
        pass
    
    def check_if_3D_points_is_in_frustum(self, 
                                        point_3d: np.array) -> bool:
        """
        Check if 3D point is in front of camera
        """
        pass