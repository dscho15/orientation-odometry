import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

def read_calib_kitty(calib_path: Path) -> list[np.ndarray]:
    with open(calib_path, "r") as f:
        lines = [list(map(float, line.strip().split(" ")[1:])) for line in f.readlines()]
    camera_matrices = [np.array(line).reshape(3, 4) for line in lines]
    return camera_matrices

class KittyDataset:
    
    def __init__(self, 
                 p_imgs: str,
                 calib: str = None,
                 timestamps: str = None,
                 grayscale: bool = True
                 ) -> None:
        
        assert Path(p_imgs).exists(), "Dataset path does not exist"
        self.p_imgs = p_imgs
        self.camera_matrices = read_calib_kitty(calib)
        self.idx = int(p_imgs.split("_")[-1])
        self.timestamps = pd.read_csv(timestamps, header=None, names=["timestamp"])
        self.imgs = sorted(list(Path(self.p_imgs).glob("*.png")))
        self.imgs = [str(img) for img in self.imgs]
        self.grayscale = grayscale
    
    @property
    def get_timestamps(self) -> pd.DataFrame:
        return self.timestamps
    
    @property
    def get_camera_matrices(self) -> list[np.ndarray]:
        return self.camera_matrices
    
    @property
    def get_camera_matrix(self) -> np.ndarray:
        return self.camera_matrices[self.idx]
    
    @property
    def get_raw_extrinsics(self) -> np.ndarray:
        cam_mat = self.camera_matrices[self.idx]
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
    
    
    

if __name__ == "__main__":
    
    kitty_dataset = KittyDataset("datasets/kitty/00/image_2", 
            "datasets/kitty/00/calib.txt", 
            "datasets/kitty/00/times.txt")
    
    print(kitty_dataset.get_camera_matrices)
    print(kitty_dataset.get_timestamps)
    kitty_dataset[0]