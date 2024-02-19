import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from pprint import pprint


def read_calib_kitty(calib_path: Path) -> list[np.ndarray]:
    
    with open(calib_path, "r") as f:
        lines = [list(line.strip().split(" ")) for line in f.readlines()]
    
    calib_data = {
        line[0]: line[1:] for line in lines
    }

    calib_data = {
        k: [float(v) for v in val] for k, val in calib_data.items()
    }

    calib_data = {
        k: np.array(P).reshape((3, 4)) if len(P) == 12 else P
        for k, P in calib_data.items()
    }

    calib_data = {
        k: np.array(R).reshape((3, 3)) if len(R) == 9 else R
        for k, R in calib_data.items()
    }

    return calib_data


class KittyDataset:
    def __init__(
        self,
        p_imgs: str,
        calib_path: str = None,
        grayscale: bool = False,
    ) -> None:
        assert Path(p_imgs).exists(), "Dataset path does not exist"
        self.p_imgs = p_imgs
        self.calib_data = read_calib_kitty(calib_path)
        self.camera_matrix = self.calib_data[f"K_0{self.p_imgs[-1]}:"]
        self.idx = int(p_imgs.split("_")[-1])
        self.imgs = Path(self.p_imgs).rglob("*.png")
        self.imgs = sorted([str(img) for img in self.imgs])
        self.grayscale = grayscale

    @property
    def get_raw_extrinsics(self) -> np.ndarray:
        cam_mat = self.camera_matrix
        fx, fy, cx, cy = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]
        return fx, fy, cx, cy

    @property
    def get_hw(self) -> tuple:
        img = Image.open(self.imgs[0])
        h, w = img.size
        return h, w

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        if self.grayscale:
            img = img.convert("L")
        img = np.array(img)
        return img

class CustomDataset:
    def __init__(
        self,
        p_imgs: str,
        pinhole_params: list[float] = None,
        fov_w_deg: float = 65,
        grayscale: bool = True,
        max_img_size: int = 1024,
        n_imgs: int = None,
    ) -> None:
        assert Path(p_imgs).exists(), "Dataset path does not exist"
        self.p_imgs = p_imgs

        self.imgs = Path(self.p_imgs).rglob("*.jpg")
        self.imgs = sorted([str(img) for img in self.imgs])
        if n_imgs is not None:
            self.imgs = self.imgs[:n_imgs]
        
        self.max_img_size = max_img_size
        self.camera_matrix = np.ones((3, 3))
        if pinhole_params is not None:
            self.camera_matrix[0, 0], self.camera_matrix[1, 1] = pinhole_params[:2]
            self.camera_matrix[0, 2], self.camera_matrix[1, 2] = pinhole_params[2:]
        elif fov_w_deg is not None:
            self.field_of_view = fov_w_deg
            
            w, h = self.get_hw
            
            if max_img_size < max(w, h):
                if h > w:
                    w = int(w * max_img_size / h)
                    h = max_img_size
                else:
                    h = int(h * max_img_size / w)
                    w = max_img_size
                
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
    
    def resize(self, img: np.ndarray) -> np.ndarray:
        (h, w) = img.shape[:2]
        
        if self.max_img_size > max(w, h):
            return img
        
        if h > w:
            w = int(w * 1024 / h)
            h = 1024
        else:
            h = int(h * 1024 / w)
            w = 1024
            
        img = Image.fromarray(img)
        img = img.resize((w, h))
        
        return np.array(img)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        
        if self.grayscale:
            img = img.convert("L")
            
        return self.resize(np.array(img))