import numpy as np
from matplotlib import pyplot as plt
import cv2
import logging
from typing import Optional


def visualize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    kps1: np.ndarray,
    kps2: np.ndarray,
    draw: int = 500,
):
    """
    Visualize a pair of images with matched features
    """
    kps1 = kps1.astype(int)
    kps2 = kps2.astype(int) + np.array([img1.shape[1], 0])

    img = np.hstack([img1, img2])

    # randomly permute kps1 and kps2
    idxs = np.random.permutation(kps1.shape[0])
    kps1 = kps1[idxs]
    kps2 = kps2[idxs]

    cnt = 0
    for kp1, kp2 in zip(kps1, kps2):
        img = cv2.line(img, tuple(kp1), tuple(kp2), (0, 255, 0), 1)
        img = cv2.circle(img, tuple(kp1), 2, (0, 0, 255), 1)
        img = cv2.circle(img, tuple(kp2), 2, (0, 0, 255), 1)
        if cnt > draw:
            break
        cnt += 1

    plt.imshow(img)
    plt.show()
    return img
