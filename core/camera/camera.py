import numpy as np
import cv2
from typing import Optional


class PinholeCamera:
    def __init__(
        self,
        height: int,
        width: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_params: Optional[np.ndarray] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.intrinsics = np.array([[fx, 0, cx], 
                                    [0, fy, cy], 
                                    [0, 0, 1]])
        self.dist_params = dist_params
        print(self.intrinsics)

    @property
    def extended_intrinsics(self) -> np.ndarray:
        return np.concatenate([self.intrinsics, np.zeros((3, 1))], axis=1)

    def dehomogenize(self, points: np.ndarray) -> np.ndarray:
        return points[:, :2] / points[:, 2, None]

    def project(self, points_3d: np.ndarray, homogeneous: bool = False) -> np.ndarray:
        points_3d = np.concatenate(
            [points_3d, np.ones((points_3d.shape[0], 1))], axis=1
        )
        points_2d = np.matmul(self.extended_intrinsics, points_3d.T).T
        if homogeneous:
            return points_2d
        return self.dehomogenize(points_2d)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self.dist_params is not None:
            undistorted_image = cv2.undistort(image, self.intrinsics, self.dist_params)
        else:
            undistorted_image = image
        return undistorted_image

    def check_if_3d_point_is_in_frustum(self, point_3d: np.ndarray) -> np.ndarray:
        homogeneous_2d_points = self.project(point_3d, True)
        return homogeneous_2d_points[:, 2] > 0
