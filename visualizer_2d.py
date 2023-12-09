import numpy as np
from matplotlib import pyplot as plt
import cv2
import logging


def visualize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    matches: np.ndarray,
    draw: int = 250,
):
    """
    Visualize a pair of images with matched features
    """
    kp1 = kp1.astype(int)
    kp2 = kp2.astype(int) + np.array([img1.shape[1], 0])

    img = np.hstack([img1, img2])

    for i, m in enumerate(matches):
        kp1_ = kp1[m[0]]
        kp2_ = kp2[m[1]]
        img = cv2.line(img, tuple(kp1_), tuple(kp2_), (0, 255, 0), 1)
        img = cv2.circle(img, tuple(kp1_), 2, (0, 0, 255), 1)
        img = cv2.circle(img, tuple(kp2_), 2, (0, 0, 255), 1)

        if i > draw:
            break

    logging.info(f"number of matches: {len(matches)}")

    plt.imshow(img, cmap="gray")
    plt.show()

    return img
