import numpy as np
from pathlib import Path

class Dataset:
    
    def __init__(self, p_datadir) -> None:
        
        assert Path(p_datadir).exists(), "Dataset path does not exist"
        self.datadir = p_datadir
        self.dataset = []
        # load dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]