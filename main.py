from argparse import ArgumentParser
from dataset import CustomDataset
from misc import get_extractor, get_matcher
from pathlib import Path
from tqdm import tqdm
from visual_odometry import VisualOdometry, PinholeCamera
from visualizer_2d import visualize_pair
from scipy.spatial.transform import Rotation
import cv2

import numpy as np
import pandas as pd

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
    m = matcher(desc1, desc2, kps1, kps2)
    n_matches = len(m) / len(kps1)
    error = kps1[m[:, 0]] - kps3[m[:, 1]]
    error = np.sqrt(np.linalg.norm(error, axis=1, ord=2))
    error = np.median(error)
    return m, n_matches, error

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
        default="/mnt/nvme0/datasets/video_blue_sky/videoes_from_michael/vid7/FRAMES/C0042",
    )
    args.add_argument("--alpha", type=float, default=0.5)
    args.add_argument("--camera_model", type=str, default="pinhole")
    args.add_argument("--feature_extractor", type=str, default="root_sift")
    args.add_argument("--matcher", type=str, default="standard")
    args.add_argument("--n_features", type=int, default=20000)
    args.add_argument("--n_ransac_max_iters", type=int, default=100000)
    args.add_argument("--ransac_confidence", type=float, default=0.99999)
    args.add_argument("--ransac_reproj_threshold", type=float, default=3.0)
    args.add_argument("--output_dir", type=str, default="output")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("-fov", "--fov_w_deg", type=float, default=75.0)
    args.add_argument("-n", "--n_imgs", type=int, default=-1)
    args.add_argument("-v", "--visualize", type=bool, default=True)

    return args.parse_known_args()[0]

            
# update last_matched_keypoint to 

# H, mask = cv2.findHomography(self.keyframe_kps[matches[:, 0]], self.cur_kps[matches[:, 1]], 0, ransacReprojThreshold=0.25, confidence=0.9999)            
# new_img = cv2.warpPerspective(self.cur_img, np.linalg.inv(H), (self.keyframe_img.shape[1], self.keyframe_img.shape[0]))
# new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

# cv2.imwrite(f'output/homographies/{i:04}.png', new_img)   
 
args = parse_args()

np.random.seed(args.seed)

dataset = CustomDataset(args.images_dir, n_imgs=args.n_imgs, fov_w_deg=args.fov_w_deg)
max_img_size = dataset.max_img_size

extractor = get_extractor(args.feature_extractor, 
                            args.n_features)

matcher = get_matcher(args.matcher,
                      args.n_ransac_max_iters,
                      args.ransac_confidence,
                      args.ransac_reproj_threshold)

camera = PinholeCamera(*dataset.get_hw, *dataset.get_raw_extrinsics)

pbar = tqdm(range(len(dataset)))

features = []
if True:
    imgs = []
for i in pbar:
    img = dataset[i]
    features.append(extractor(img))
    if True:
        imgs.append(img)

connections = []
potential_loops = []
start = 0
offset = 10
(h, w) = dataset.get_hw
idx = 0

for i in range(1, len(features) - offset):
    
    (kps1, desc1) = features[i - 1]
    (kps2, desc2) = features[i]
    (kps3, desc3) = features[i + offset]

    m13, n_matches13, error13 = compute_error_and_median(matcher, desc1, desc3, kps1, kps3)
    m23, n_matches23, error23 = compute_error_and_median(matcher, desc2, desc3, kps2, kps3)
    
    if True:
        visualize_pair(
            imgs[i-1],
            imgs[i+offset],
            kps1,
            kps3,
            m13,
            args.output_dir,
            idx=idx
        )
        idx += 1
    
    connections.append((i-1, i, i + offset, m13, m23))

    print(f"Frame {i-1} | {i} | {i + offset} - {n_matches13:.3f} | {n_matches23:.3f} matches - {error13:.3f} | {error23:.3f} px movement")

rotm_pairs = []
def compute_pose(kps1, kps2, m, camera):
    F = cv2.findFundamentalMat(kps1[m[:, 0]], kps2[m[:, 1]], cv2.FM_LMEDS)[0]
    E = camera.K.T @ F @ camera.K
    _, R, t, _ = cv2.recoverPose(E, kps1[m[:, 0]], kps2[m[:, 1]], focal=1, pp=(0., 0.))
    return R, t, E

for (i, j, k, mik, mjk) in tqdm(connections[1:]):
    
    kpsi = features[i][0]
    kpsj = features[j][0]
    kpsk = features[k][0]
    
    Rik, Eik, tik = compute_pose(kpsi, kpsk, mik, camera) 
    Rjk, Ejk, tjk = compute_pose(kpsj, kpsk, mjk, camera)
    
    rotm_pairs.append((Rik, Rjk))
    
ori_traj_new = [np.eye(3, 3)]
for (Rik, Rjk) in rotm_pairs:
    ori_traj_new.append(ori_traj_new[-1] @ (Rik.T @ Rjk))
            
quats = convert_poses_to_quaternions(ori_traj_new)
lp_filtered = lp_filter(quats, alpha=args.alpha, normalize=False)
lp_filtered = lp_filter(lp_filtered[::-1], alpha=args.alpha, normalize=True)[::-1]

orig_data = pd.DataFrame(quats, columns=['x', 'y', 'z', 'w'])
orig_data['img'] = dataset.imgs[:len(quats)]
orig_data.to_csv(f'output/{Path(args.images_dir).stem}.csv', index=False)
