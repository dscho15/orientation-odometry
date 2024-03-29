import numpy as np
from pathlib import Path
from PIL import Image

class CustomDataset:
    def __init__(
        self,
        p_imgs: str,
        pinhole_params: list[float] = None,
        fov_w_deg: float = 70,
        grayscale: bool = False,
        n_imgs: int = None,
    ) -> None:

        assert Path(p_imgs).exists(), "Dataset path does not exist"
        self.p_imgs = p_imgs

        self.imgs = Path(self.p_imgs).rglob("*.jpg")
        self.imgs = sorted([str(img) for img in self.imgs])

        if n_imgs is not None:
            self.imgs = self.imgs[:n_imgs]

        self.camera_matrix = np.ones((3, 3))

        if pinhole_params is not None:
            
            self.camera_matrix[0, 0], self.camera_matrix[1, 1] = pinhole_params[:2]
            self.camera_matrix[0, 2], self.camera_matrix[1, 2] = pinhole_params[2:]

        elif fov_w_deg is not None:
            
            self.field_of_view = fov_w_deg
            h, w = self.get_hw

            self.camera_matrix[0, 0] = w / (2 * np.tan(np.radians(fov_w_deg) / 2))
            self.camera_matrix[1, 1] = w / (2 * np.tan(np.radians(fov_w_deg) / 2))
            self.camera_matrix[0, 2] = w / 2
            self.camera_matrix[1, 2] = h / 2

        self.grayscale = grayscale

    @property
    def get_raw_extrinsics(self) -> np.ndarray:
        cam_mat = self.camera_matrix
        fx, fy, cx, cy = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]
        return fx, fy, cx, cy

    @property
    def get_hw(self) -> tuple:
        img = Image.open(self.imgs[0])
        w, h = img.size
        return h, w

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])

        if self.grayscale:
            img = img.convert("L")

        return np.array(img)
