import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path

def visualize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    kps1: np.ndarray,
    kps2: np.ndarray,
    matches: np.ndarray,
    output_dir: str,
    mask: np.ndarray = None,
    draw: int = 5000,
    flow: bool = True,
    idx: int = 0,
):    
    kps1 = kps1.astype(int)
    kps2 = kps2.astype(int)
    
    if not flow:
        kps2 += np.array([img1.shape[1], 0])

    if not flow:
        img = np.hstack([img1, img2])
    else:
        img = img2
        
    indices = matches
    kps1 = kps1[indices[:, 0]]
    kps2 = kps2[indices[:, 1]]

    if mask is not None:
        kps1 = kps1[mask.ravel() == 1]
        kps2 = kps2[mask.ravel() == 1]
    
    idxs = np.random.permutation(kps1.shape[0])
    kps1 = kps1[idxs]
    kps2 = kps2[idxs]
    
    cnt = 0
    for kp1, kp2 in zip(kps1, kps2):
        img = cv2.arrowedLine(img, tuple(kp1), tuple(kp2), (0, 255, 0), 1)
        if cnt > draw:
            break
        cnt += 1
    
    output_dir = Path(output_dir) / "vis_matches"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(f"{str(output_dir)}/img_{idx}.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    return img