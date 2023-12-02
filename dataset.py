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
                 p_datadir: str,
                 calib: str = None,
                 timestamps: str = None,
                 grayscale: bool = True
                 ) -> None:
        
        assert Path(p_datadir).exists(), "Dataset path does not exist"
        self.datadir = p_datadir
        self.camera_matrices = read_calib_kitty(calib)
        self.idx = int(p_datadir.split("_")[-1])
        self.timestamps = pd.read_csv(timestamps, header=None, names=["timestamp"])
        self.imgs = sorted(list(Path(self.datadir).glob("*.png")))
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
        
    def __len__(self):
        return len(self.dataset)
    
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