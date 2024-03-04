from argparse import ArgumentParser
from dataset import CustomDataset
from misc import get_extractor, get_matcher
from pathlib import Path
from tqdm import tqdm
from visualizer_2d import visualize_pair
from scipy.spatial.transform import Rotation
import cv2
from misc import load_imgs_using_multithreading
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class PinholeCamera(object):
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def convert_poses_to_quaternions(
    ori_traj: list[np.ndarray],
):
    rotm_matrices = np.array([rotm for rotm in ori_traj])
    quaternions = np.array([Rotation.from_matrix(rotm).as_quat() for rotm in rotm_matrices])  
    return quaternions

def lp_filter(x: np.ndarray, alpha: float, normalize=True):
    
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    y = np.zeros_like(x)
    y[0, :] = x[0, :]
    for i in range(1, len(x)):
        y[i, :] = alpha * y[i-1, :] + (1 - alpha) * x[i, :]
    for j in range(len(x[0])):
        if normalize:
            y[j, :] = y[j, :] / np.linalg.norm(y[j, :])
    return y

def compute_error_and_median(matcher, desc1, desc2, kps1, kps2):
    matches = matcher(desc1, desc2, kps1, kps2)
    n_matches = len(matches) / len(kps1)
    error = kps1[matches[:, 0]] - kps2[matches[:, 1]]
    error = np.sqrt(np.linalg.norm(error, axis=1, ord=2))
    error = np.median(error)
    return matches, n_matches, error

def parse_args():
    args = ArgumentParser(description="Visual Odometry")
    args.add_argument(
        "-d",
        "--dataset",
        default="custom",
        type=str
    )
    args.add_argument(
        "--images_dir",
        type=str,
        default="/home/daniel/Desktop/pyessential-solver/FRAMES/C0042",
    )
    args.add_argument("--alpha", type=float, default=0.90)
    args.add_argument("--camera_model", type=str, default="pinhole")
    args.add_argument("--feature_extractor", type=str, default="orb")
    args.add_argument("--matcher", type=str, default="standard")
    args.add_argument("--n_features", type=int, default=15000)
    args.add_argument("--output_dir", type=str, default="/home/daniel/Desktop/orientation-odometry/output")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("-fov", "--fov_w_deg", type=float, default=85.0)
    args.add_argument("-n", "--n_imgs", type=int, default=-1)
    args.add_argument("-v", "--visualize", type=bool, default=True)

    return args.parse_known_args()[0]

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)

    dataset = CustomDataset(args.images_dir, n_imgs=args.n_imgs, fov_w_deg=args.fov_w_deg)

    extractor = get_extractor(args.feature_extractor, 
                              args.n_features)

    matcher = get_matcher(args.matcher)

    camera = PinholeCamera(*dataset.get_raw_extrinsics)

    pbar = tqdm(range(len(dataset)))
    
    imgs = load_imgs_using_multithreading(dataset.imgs)

    features = []
    for img in tqdm(imgs):
        features.append(extractor(img))

    connections = []
    (h, w) = dataset.get_hw
    for i in tqdm(range(1, len(features))):
        (kps1, desc1) = features[i - 1]
        (kps2, desc2) = features[i]
        matches, ratio_matches, error = compute_error_and_median(matcher, desc1, desc2, kps1, kps2)
        connections.append(matches)

    rotm_pairs = []
    def compute_pose(kps1, kps2, camera):
        
        F, mask = cv2.findFundamentalMat(kps1, kps2, cv2.FM_LMEDS)
        
        kps1 = kps1[mask.ravel() == 1]
        kps2 = kps2[mask.ravel() == 1]
        
        E = camera.K.T @ F @ camera.K
        
        _, R, t, _ = cv2.recoverPose(E, kps1, kps2, focal=1, pp=(0., 0.))
        
        return R, t, E, kps1, kps2

    rotms = []
    for i in tqdm(range(1, len(connections))):
        
        matches = connections[i-1]
        indices1, indices2 = matches[:, 0], matches[:, 1]
        
        kps1 = features[i-1][0][indices1]
        kps2 = features[i][0][indices2]
        
        R, E, t, kp1, kp2 = compute_pose(kps1, kps2, camera)
        
        visualize_pair(imgs[i], imgs[i+1], kp1, kp2, None, args.output_dir, idx=i)
        
        angle = cv2.Rodrigues(R)[0]
        angle = cv2.norm(angle, cv2.NORM_L2)
        
        if angle > np.pi / 8:
            rotms.append(np.eye(3, 3))
        else:
            rotms.append(R)
            
    ori_traj_new = [np.eye(3, 3)]
    for (R) in rotms:
        ori_traj_new.append(ori_traj_new[-1] @ R)
        
    quats = convert_poses_to_quaternions(ori_traj_new)
    lp_filtered = lp_filter(quats, alpha=args.alpha, normalize=False)
    lp_filtered = lp_filter(lp_filtered[::-1], alpha=args.alpha, normalize=True)[::-1]

    orig_data = pd.DataFrame(quats, columns=['x', 'y', 'z', 'w'])
    orig_data['img'] = dataset.imgs[:len(quats)]
    orig_data.to_csv(f'output/{Path(args.images_dir).stem}.csv', index=False)

    plt.subplot(411)
    plt.plot(quats[:, 0])
    plt.title('x')
    plt.subplot(412)
    plt.plot(quats[:, 1])
    plt.title('y')
    plt.subplot(413)
    plt.plot(quats[:, 2])
    plt.title('z')
    plt.subplot(414)
    plt.plot(quats[:, 3])
    plt.title('w')
    plt.show()
