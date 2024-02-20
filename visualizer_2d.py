import numpy as np
from matplotlib import pyplot as plt
import cv2
import os


def visualize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    kps1: np.ndarray,
    kps2: np.ndarray,
    draw: int = 500,
    flow: bool = True
):
    """
    Visualize a pair of images with matched features
    """
    kps1 = kps1.astype(int)
    kps2 = kps2.astype(int)
    if not flow:
        kps2 += np.array([img1.shape[1], 0])

    if not flow:
        img = np.hstack([img1, img2])
    else:
        img = img2

    # randomly permute kps1 and kps2
    idxs = np.random.permutation(kps1.shape[0])
    kps1 = kps1[idxs]
    kps2 = kps2[idxs]
    
    # visualize the flow of the featuree, i.e. the lines between the matched features in a single image
    cnt = 0
    for kp1, kp2 in zip(kps1, kps2):
        img = cv2.line(img, tuple(kp1), tuple(kp2), (0, 255, 0), 1)
        if cnt > draw:
            break
        cnt += 1
    
    file_path = "/home/dts/colmap/visual-odometry/output/imgs"
    files = os.listdir(file_path)
    
    cv2.imwrite(f"{file_path}/img_{len(files)}.jpg", img)

    return img