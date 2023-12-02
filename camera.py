import numpy as np
import cv2

class PinholeCamera:
    
    def __init__(self,
                 h: int, 
                 w: int,
                 cx: float,
                 cy: float,
                 fx: float,
                 fy: float,
                 dist_params: np.array = None) -> None:
        self.w = w
        self.h = h
        self.intrinsics = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
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