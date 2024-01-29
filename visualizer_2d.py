import numpy as np
from matplotlib import pyplot as plt
import cv2


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

def visualize_quaternion_trajectory(
    odom: list[np.ndarray],
):
    rotm = np.array([pose[:3, :3] for pose in odom])

    def rotm_2_quat(rotm):
        qw = np.sqrt(1 + rotm[0, 0] + rotm[1, 1] + rotm[2, 2]) / 2
        if qw < 0:
            qw = -qw
        qx = (rotm[2, 1] - rotm[1, 2]) / (4 * qw)
        qy = (rotm[0, 2] - rotm[2, 0]) / (4 * qw)
        qz = (rotm[1, 0] - rotm[0, 1]) / (4 * qw)
        return np.array([qw, qx, qy, qz])
    
    quats = np.array([rotm_2_quat(rotm) for rotm in rotm])
    
    # plt.plot(quats[:, 0], label="qw")
    # plt.plot(quats[:, 1], label="qx")
    # plt.plot(quats[:, 2], label="qy")
    # plt.plot(quats[:, 3], label="qz")
    # plt.show()

    return quats